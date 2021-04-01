import argparse
from sys import prefix
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import torch.optim as optim
import torch.utils.data as data_utils
import time
from torchvision.utils import save_image
import matplotlib.pyplot as plt
# from gym_unity.envs import UnityToGymWrapper
import yaml

import numpy as np
# from DDPG.DDPG import DDPG
from Utils.Helpers import NaiveModel, MultiAgentUnityEnv, ids2indexes, update_obs_list, slice_obs_list, HumanOperator
from Utils.ReplayBuffer import ReplayBuffer
from Utils.ReplayBufferM import ReplayBufferM
from unity_comm.StringLogChannel import StringLogChannel
import torch
import math
from TD3.TD3 import TD3
# from TD3.utils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import cv2

# Runs model policy for X episodes
def eval_model(model, env, eval_episodes=10, rec_arch=False, verbose=False, parameters=None, render = False):
    episodic_reward_list = []
    success_rate_list = []
    episodic_length_list = []
    episodic_reward = np.zeros(env.no_of_agents)
    agent_timesteps = np.zeros(env.no_of_agents)
    agent_no_episodes = np.zeros(env.no_of_agents)
    success_reward_value = parameters["simulation"]["on_hit_target"]

    (agents_action, agents_prev_state, step_actions, prev_state_was_terminal, step_prev_states, step_agents) = env.reset()

    e = 0.
    no_steps = 0
    if rec_arch:
        actor_hidden_states, actor_cell_states = model.actor.gen_empty_h_states(env.no_of_agents, model.device)
        critic_hidden_states, critic_cell_states = model.critic.gen_empty_h_states(env.no_of_agents, model.device)
    while(e < eval_episodes):
        # Select action according to policy
        if(step_agents.size != 0):
            if rec_arch:
                step_actions, (actor_hidden_states[:, step_agents, :], actor_cell_states[:, step_agents, :]), (critic_hidden_states[:, step_agents, :], critic_cell_states[:, step_agents, :]) = model.react_to_obs(slice_obs_list(agents_prev_state, step_agents), ((actor_hidden_states[:, step_agents, :], actor_cell_states[:, step_agents, :]), (critic_hidden_states[:, step_agents, :], critic_cell_states[:, step_agents, :])), add_noise=False)
            else:
                # step_actions = env.action_space_sample(step_agents.size)
                step_actions = model.react_to_obs(slice_obs_list(agents_prev_state, step_agents), add_noise=False)                               
            update_obs_list(obs_list = agents_prev_state, new_obs = step_prev_states, ids = step_agents)
        
        if render:
            image = agents_prev_state[0][0][0]
            image = image*255
            image = image.astype(np.uint8)                
            image = cv2.resize(image, (256, 256)) 
            cv2.imshow('Agent image',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break             

        decision, terminals = env.step(step_actions)
        step_agents = np.array([])
        step_actions = np.array([])

        if (terminals != None):
            tstate, treward, tdone, tagents_ids = terminals
            prev_state_was_terminal[tagents_ids] = True
            no_steps += treward.shape[0]
            if verbose:
                print("Episode ended for agents ", end="")
                print(tagents_ids)
            for id, reward in zip(tagents_ids, treward):
                episodic_reward[id] += reward
                agent_timesteps[id] += 1.0
                episodic_reward_list.append(episodic_reward[id])
                episodic_length_list.append(agent_timesteps[id])
                if(reward == success_reward_value):
                    success_rate_list.append(1.0)
                    success_status = "success"
                else:
                    success_rate_list.append(0.0)
                    success_status = "failure"
                # cumulative_success_rate[tagents_ids[success_agents]] +=1.0
                if verbose:
                    print("\tAgent (%d) episode reward: %.2f, timesteps: %d. Episode was a %s" % (id, episodic_reward[id], agent_timesteps[id], success_status))
                agent_no_episodes[id] += 1.0
                # print("Agent %d ended with reward %f and timesteps %f. Curr no epi %d" % (id, reward, agent_timesteps[id], agent_no_episodes[id]))
                episodic_reward[id] = 0.
                agent_timesteps[id] = 0.
                e += 1.0
                # quit()
                # print(e)
            if(rec_arch):
                    # Clearing actor/critic hidden and cell states
                    actor_cell_states[:, tagents_ids, :] = torch.zeros((1, tagents_ids.shape[0], model.actor.lstm_hidden_dim)).to(model.device)
                    actor_hidden_states[:, tagents_ids, :] = torch.zeros((1, tagents_ids.shape[0], model.actor.lstm_hidden_dim)).to(model.device)
                    critic_cell_states[:, tagents_ids, :] = torch.zeros((2, tagents_ids.shape[0], model.critic.lstm_hidden_dim)).to(model.device)
                    critic_hidden_states[:, tagents_ids, :] = torch.zeros((2, tagents_ids.shape[0], model.critic.lstm_hidden_dim)).to(model.device)

        if (decision != None):
            dstate, dreward, ddone, dagents_ids = decision
            no_steps += dreward.shape[0]
            # IDS that will BE be stored in buffer.
            step_agents = dagents_ids.copy()
            step_prev_states = dstate
            for id, reward in zip(dagents_ids, dreward):
                episodic_reward[id] += reward
                agent_timesteps[id] += 1.0
                # if(verbose):
                #     print("Agent %d reward's %f and timesteps %f. Curr no epi %d" % (id, reward, agent_timesteps[id], agent_no_episodes[id]))
        # print("##################")
 
    if render:
        cv2.destroyAllWindows()
    return (np.mean(episodic_reward_list), np.std(episodic_reward_list)), (np.mean(episodic_length_list), np.std(episodic_length_list)), (np.mean(success_rate_list), np.std(success_rate_list)), no_steps

def train_model(model, env, replay_buffer, string_log, buffer_size_to_train=25e3, eval_freq = 5000, number_of_eval_episodes = 100, max_steps=1000000, save_best=True, render=False, writer=None, buffer_op="generate", curriculum=False, augmentation_indexes = None, use_memory=False, step_update_ratio=1, parameters=None):
    if(augmentation_indexes != None):
        use_augmentation = True
        augmentation_indexes = np.array(augmentation_indexes)
    else:
        use_augmentation = False

    best_model_reward = -9999999
    # Collecting experience to initialize the replay buffer
    if(not curriculum):
        print("Collecting %d experiences based on a random policy." % (buffer_size_to_train))
    else:
        print("Collecting %d experiences based on a previously trained policy." % (buffer_size_to_train))
    start_b_time = time.time()

    t = 0
    prev_t = 0
    total_model_updates = 0
    total_evaluation_steps = 0
    total_evaluation_time = 0
    no_episodes = 0

    (agents_action, agents_prev_state, step_actions, prev_state_was_terminal, step_prev_states, step_agents) = env.reset()
    next_report = buffer_size_to_train/10.0 # Will try to print info in 10% intervals

    collecting_experiences = True
    training = False
    if use_memory and curriculum:
        no_random_obs = env.no_of_agents * replay_buffer.memory_length * 2
        print("Collecting %d observations to prevent buffer error" % no_random_obs)
    else:
        no_random_obs = 0
    if(use_memory):
        actor_hidden_states, actor_cell_states = model.actor.gen_empty_h_states(env.no_of_agents, replay_buffer.device)
        critic_hidden_states, critic_cell_states = model.critic.gen_empty_h_states(env.no_of_agents, replay_buffer.device)
    try:
        while (collecting_experiences or training):
            if collecting_experiences and (t >= buffer_size_to_train):
                # Finished collecting experiences, preparing training...

                print("Done collecting experiences (took %.2f seconds). Training..." % (time.time() - start_b_time))                

                collecting_experiences = False
                training = True
                t = 1
                prev_t = 1
                steps_since_last_model_update = 0
                parcial_no_of_model_updates = 0

                (agents_action, agents_prev_state, step_actions, prev_state_was_terminal, step_prev_states, step_agents) = env.reset()

                next_report = eval_freq
                training_time_start = time.time()
                if(use_memory):
                    actor_hidden_states, actor_cell_states = model.actor.gen_empty_h_states(env.no_of_agents, replay_buffer.device)
                    critic_hidden_states, critic_cell_states = model.critic.gen_empty_h_states(env.no_of_agents, replay_buffer.device)
                    replay_buffer.reset_seq_ptr()
            if training and (t >= max_steps):
                # Finished training, exiting training loop...
                print("Done training. Saving model and exiting...")
                training = False
                break

            if(step_agents.size != 0):
                # if (collecting_experiences and (not curriculum)) or (curriculum and (random_samples > 0)):
                if (collecting_experiences and (not curriculum)) or (no_random_obs > 0):
                    # Select action randomly
                    step_actions = env.action_space_sample(step_agents.size)
                    if(no_random_obs > 0):
                        no_random_obs-=1
                else:
                    # Select based on trained model
                    if use_memory:
                        step_actions, (actor_hidden_states[:, step_agents, :], actor_cell_states[:, step_agents, :]), (critic_hidden_states[:, step_agents, :], critic_cell_states[:, step_agents, :]) = model.react_to_obs(slice_obs_list(agents_prev_state, step_agents), ((actor_hidden_states[:, step_agents, :], actor_cell_states[:, step_agents, :]), (critic_hidden_states[:, step_agents, :], critic_cell_states[:, step_agents, :])), t_device = replay_buffer.device, add_noise=True)
                    else:
                        step_actions = model.react_to_obs(slice_obs_list(agents_prev_state, step_agents), add_noise=True)
                    step_actions = step_actions.cpu().data.numpy()
                update_obs_list(obs_list = agents_prev_state, new_obs = step_prev_states, ids = step_agents)
                agents_action[step_agents] = step_actions

            # Performing actions for the requesting agents
            decision, terminals = env.step(step_actions)
            prev_t = t
            step_agents = np.array([])
            step_actions = np.array([])
            # Processing terminal and non-terminal transitions
            if (terminals != None):
                # For terminal states, we need to:
                # - Store the terminal transition
                tstate, treward, tdone, tagents_ids = terminals
                terminal_prev_states = slice_obs_list(agents_prev_state, tagents_ids)
                if use_memory:
                    replay_buffer.record(terminal_prev_states, agents_action[tagents_ids], treward, tstate, tdone, actor_cell_states[:, tagents_ids], critic_cell_states[:, tagents_ids], tagents_ids)
                else:
                    replay_buffer.record(terminal_prev_states, agents_action[tagents_ids], treward, tstate, tdone, tagents_ids)
                t += treward.shape[0]
                no_episodes += treward.shape[0]
                prev_state_was_terminal[tagents_ids] = True

                if(use_memory):
                    # Clearing actor/critic hidden and cell states
                    actor_cell_states[:, tagents_ids, :] = torch.zeros((1, tagents_ids.shape[0], model.actor.lstm_hidden_dim)).to(replay_buffer.device)
                    actor_hidden_states[:, tagents_ids, :] = torch.zeros((1, tagents_ids.shape[0], model.actor.lstm_hidden_dim)).to(replay_buffer.device)
                    critic_cell_states[:, tagents_ids, :] = torch.zeros((2, tagents_ids.shape[0], model.critic.lstm_hidden_dim)).to(replay_buffer.device)
                    critic_hidden_states[:, tagents_ids, :] = torch.zeros((2, tagents_ids.shape[0], model.critic.lstm_hidden_dim)).to(replay_buffer.device)

            if (decision != None):
                # We will only store in the memory buffer transitions made by agents whose last state was not
                # a terminal state, i.e. , agents in witch the current state is NOT a first state
                dstate, dreward, ddone, dagents_ids = decision

                # IDS that will BE be stored in buffer.
                to_buffer_agents_ids = dagents_ids[~prev_state_was_terminal[dagents_ids]]
                # We can't index dstate, dreward with the agents ids, instead we have to convert then to indexes using dagents_ids
                # e.g.:
                #   dagents_ids = [3, 10, 15, 19, 22], to_buffer_agents_ids=[3, 19, 15] will yeld:
                #       ids2indexes(dagents_ids, to_buffer_agents_ids) = [0, 3, 2]
                to_buffer_agents_indexes = ids2indexes(dagents_ids, to_buffer_agents_ids)
                if(to_buffer_agents_ids.size != 0):
                    prev_states_to_record = slice_obs_list(agents_prev_state, to_buffer_agents_ids)
                    states_to_record = slice_obs_list(dstate, to_buffer_agents_indexes)
                    if use_memory:
                        replay_buffer.record(prev_states_to_record, agents_action[to_buffer_agents_ids], dreward[to_buffer_agents_indexes], states_to_record, ddone[:to_buffer_agents_indexes.shape[0]], actor_cell_states[:, to_buffer_agents_ids], critic_cell_states[:, to_buffer_agents_ids], to_buffer_agents_ids)
                    else:
                        replay_buffer.record(prev_states_to_record, agents_action[to_buffer_agents_ids], dreward[to_buffer_agents_indexes], states_to_record, ddone[:to_buffer_agents_indexes.shape[0]], to_buffer_agents_ids)
                    t += dreward[to_buffer_agents_indexes].shape[0]

                prev_state_was_terminal[dagents_ids] = False
                step_agents = dagents_ids.copy()
                step_prev_states = dstate
            env_steps = t - prev_t

            if (training):
                steps_since_last_model_update += env_steps
                # The model will be updated one or more times
                no_of_model_updates = math.floor(steps_since_last_model_update * step_update_ratio+parcial_no_of_model_updates)
                # print("env_steps %d, no_of_model_updates %d" % (env_steps, no_of_model_updates))
                if(no_of_model_updates >= 1):
                    parcial_no_of_model_updates = steps_since_last_model_update * step_update_ratio+parcial_no_of_model_updates - no_of_model_updates
                    total_model_updates += no_of_model_updates
                    steps_since_last_model_update = 0
                    for _ in range(no_of_model_updates):
                        model.step(replay_buffer, augmented_features=augmentation_indexes, recurrent_arch=use_memory, non_visual_index= env.non_visual_obs_index)

            if(t >= next_report):
                if collecting_experiences:
                    print("Collected %d experiences (%.2fs). Current Throughput: %.0f samples/s..." % (t, time.time() - start_b_time, t/(time.time() - start_b_time)))
                    next_report += buffer_size_to_train/10.0
                elif training:
                    print("Evaluating for %d episodes... " % (number_of_eval_episodes), end="", flush=True)
                    start_time = time.time()
                    (mean_episodic_r, episodic_r_std), (mean_episodic_length, episodic_std), (mean_success, success_std), ev_steps = eval_model(model = model, env = env, eval_episodes = number_of_eval_episodes, rec_arch=use_memory, parameters = parameters)
                    total_evaluation_steps += ev_steps
                    total_evaluation_time += time.time() - start_time
                    print("Done. Took %.2f seconds (%d total evaluation steps)" % (time.time() - start_time, ev_steps))
                    print("Training step no (Est = %d, Real = %d ). Total training time inc. eval. (%ds):" % (next_report, t, time.time() - training_time_start))
                    print("\tModel updates (%d)" % (total_model_updates))
                    print("\tTotal training episodes (%d)" % (no_episodes))                    
                    if writer is not None:
                        print("\tMean reward %f (± %f)" % (mean_episodic_r, episodic_r_std))
                        print("\tMean success rate %.2f%% (± %.2f%%)" % (mean_success*100, success_std*100))
                        print("\tActor loss %f" % (model.a_loss/eval_freq))
                        print("\tCritic loss %f" % (model.c_loss/eval_freq))
                        if use_augmentation:
                            print("\tAugmented function Actor loss %f" % (model.aug_a_loss/eval_freq))
                            print("\tAugmented function Critic loss %f" % (model.aug_c_loss/eval_freq))
                        print("\tMean episode length %f (± %f)" % (mean_episodic_length, episodic_std))
                        writer.add_scalar('model/Actor loss', model.a_loss/eval_freq, next_report)
                        writer.add_scalar('model/Critic loss', model.c_loss/eval_freq, next_report)
                        writer.add_scalar('model/Model updates', total_model_updates, next_report)
                        if use_augmentation:
                            writer.add_scalar('model/Augmented function Actor loss', model.aug_a_loss/eval_freq, next_report)
                            writer.add_scalar('model/Augmented function Critic loss', model.aug_c_loss/eval_freq, next_report)
                        writer.add_scalar('environment/Mean reward', mean_episodic_r, next_report)
                        writer.add_scalar('environment/Mean episode length', mean_episodic_length, next_report)
                        writer.add_scalar('environment/Mean Success rate', mean_success, next_report)
                        writer.flush()
                    model.a_loss = 0
                    model.c_loss = 0
                    if use_augmentation:
                        model.aug_a_loss = 0
                        model.aug_c_loss = 0
                    if save_best and (mean_episodic_r >= best_model_reward):
                        print("Saving current model... ", end = '')
                        best_model_reward = mean_episodic_r
                        model.save("./models")
                        print("Done.")
                    next_report += eval_freq

                    (agents_action, agents_prev_state, step_actions, prev_state_was_terminal, step_prev_states, step_agents) = env.reset()
                    if(use_memory):
                        # We need to reset the agents hidden states, because the environment has been reset
                        actor_hidden_states, actor_cell_states = model.actor.gen_empty_h_states(env.no_of_agents, replay_buffer.device)
                        critic_hidden_states, critic_cell_states = model.critic.gen_empty_h_states(env.no_of_agents, replay_buffer.device)
                        # Also, all the observations that were being recorded inside the current sequence need to be reset:
                        replay_buffer.reset_seq_ptr()
    except KeyboardInterrupt:
        pass
    print("#####################################")
    print("Training routine ended. Summary:")
    print("\tTotal training steps: %d" % (t))
    print("\tTotal evaluation steps: %d" % (total_evaluation_steps))
    print("\tTotal evaluation time: %ds" % (total_evaluation_time))
    print("#####################################")
    if writer is not None:
        writer.close()
    # Saving last execution:
    print("Saving last execution...")
    model.save("./models", prefix = "last_exec_")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="The run id")
    parser.add_argument("--config_file", default=None, help="The configuration file.")
    parser.add_argument("--env_location", default=None, help="The location of the environment executable. If not set connects to the editor (Default: None")
    parser.add_argument("--exec_type", default="eval", help="The execution type (Default: eval)")
    parser.add_argument("--eval_best", default="false", help="Wether to load the best model or the last saved model (Default: true)")
    parser.add_argument("--device", default="cpu", help="The device to run the model on (Default: cpu)")
    parser.add_argument("--simu_spd",default=1.0, type=float, help="The simulation speed (Default: 1.0)")
    parser.add_argument("--eval_episodes",default=-1.0, type=float, help="The simulation speed (Default: 1.0)")
    parser.add_argument("--seed",default=0, type=int, help="The number of episodes when evaluating. If -1 is passed, uses the value on the parameters file. (Default: -1)")
    parser.add_argument("--manual_control", default="false", help="Overrides the RL agent and reads input from the gamepad (Default: false)")
    parser.add_argument("--naive_policy", default="false", help="Uses a naive policy that only goes straight (Default: false)")
    parser.add_argument("--visualize_input", default="false", help="Visualize agent image input (Default: false)")

    args = parser.parse_args()
    with open(args.config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    conf_channel = EngineConfigurationChannel()
    parameter_channel = EnvironmentParametersChannel()
    string_log = StringLogChannel()
    if(args.seed != 0): 
        # This means that the used set a diferent seed in cmd
        parameters["random_seed"] = args.seed
    # if(args.simu_spd != 1.0): 
    #     # This means that the used set a diferent simulation speed in cmd
    #     parameters["time_scale"] = args.simu_spd
    
    if(args.env_location is None):
        unity_env = UnityEnvironment(side_channels=[conf_channel, string_log, parameter_channel])
    else:
        unity_env = UnityEnvironment(args.env_location, side_channels=[conf_channel, string_log, parameter_channel])
    parameter_channel.set_float_parameter("seed", parameters["random_seed"])
    env_parameters = parameters["simulation"]
    for element in env_parameters:
        parameter_channel.set_float_parameter(element, env_parameters[element])
    if(args.exec_type == "train"):
        parameter_channel.set_float_parameter("training", 1.0)
    else:
        parameters["time_scale"] = args.simu_spd
        parameter_channel.set_float_parameter("training", 0.0)

    if(args.eval_episodes != -1.0):
        parameters["eval_episodes"] = args.eval_episodes

    conf_channel.set_configuration_parameters(time_scale = parameters["time_scale"])
    parameter_channel.set_float_parameter("parameters_set", 1.0)
    
    env = MultiAgentUnityEnv(unity_env, encoder=None)

    model = None

    simu_info = {}
    print("----- ENV INFO -------")
    print(parameters["random_seed"])
    print(env.state_dim)
    print(env.action_dim)
    print(env.action_magnitude)
    print(env.no_of_agents)
    print(env.visual_obs_indexes)
    print(env.non_visual_obs_index)

    simu_info["state_dimension"] = env.state_dim
    simu_info["action_dimension"] = env.action_dim
    simu_info["action_magnitude"] = env.action_magnitude
    simu_info["no_of_agents"] = env.no_of_agents

    if(args.env_location == None):
        simu_info["env_type"] = "Editor"
    else:
        simu_info["env_type"] = args.env_location.split("/")[-1].split(".")[0]
    parameters["simu_info"] = simu_info
    print("------------")
    # quit()

    # env.seed(seed)
    torch.manual_seed(parameters["random_seed"])
    np.random.seed(parameters["random_seed"])
    rl_algorithm = parameters["rl_algorithm"]
    if "memory" in parameters:
        mem_parameters = parameters["memory"]
    else:
        mem_parameters = None
    if "augmentation" in parameters:
        aug_parameters = parameters["augmentation"]
    else:
        aug_parameters = {}
        aug_parameters["indexes"] = None
    # quit()
    if(rl_algorithm["type"] == "DDPG"):
        pass
        # model = DDPG(
        #     num_states,
        #     num_actions,
        #     model_name=args.model_name,
        #     actor_lr=1e-4,
        #     critic_lr=1e-3,
        #     device=args.device,
        #     net_config=args.net_name
        # )
    elif(rl_algorithm["type"] == "TD3"):
        kwargs = {
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            # "model_name": parameters["run_id"],
            "model_name": args.run_id,
            "max_action": env.action_magnitude,
            "net_config_name": parameters["architecture_type"],
            "device": args.device,
            "discount": rl_algorithm["discount"],
            "tau": rl_algorithm["tau"],
            "policy_noise": rl_algorithm["policy_noise"] * env.action_magnitude,
            "expl_noise": rl_algorithm["expl_noise"],
            "noise_clip": rl_algorithm["noise_clip"] * env.action_magnitude,
            "policy_freq": rl_algorithm["policy_freq"],
            "mem_parameters": mem_parameters
        }
        model = TD3(**kwargs)
        simu_info["actor_total_params"] = model.actor_total_params
        simu_info["critic_total_params"] = model.critic_total_params
    if(args.exec_type == "train"):
        rb_parameters = parameters["replay_buffer"]
        has_curriculum = parameters["base_run_id"] != "None"
        if(rb_parameters["location"] != "None"):
            rb = ReplayBuffer.load(rb_parameters["location"], device="cpu")
        else:
            if (model.actor.memory_capable() and model.critic.memory_capable()):
                rb = ReplayBufferM(
                    state_space_dim = env.state_dim,
                    action_dim = env.action_dim,
                    no_of_agents = env.no_of_agents,
                    memory_length = mem_parameters["memory_length"],
                    buffer_capacity = rb_parameters["size"],
                    batch_size = parameters["batch_size"],
                    a_lstm_hidden_dim = model.actor.lstm_hidden_dim,
                    c_lstm_hidden_dim = model.critic.lstm_hidden_dim,
                    device="cpu"
                )
            else:
                rb = ReplayBuffer(
                    env.state_dim,
                    env.action_dim,
                    rb_parameters["size"],
                    parameters["batch_size"],
                    device="cpu"
                )
        if(has_curriculum):
            model_type_str = "best" if args.eval_best == "true" else "latest"
            print("Transfering learning from a previous model. The %s model will be loaded..." % (model_type_str))
            if(args.eval_best == "true"):
                model.load("./models",name=parameters["base_run_id"], prefix="")
            else:
                model.load("./models",name=parameters["base_run_id"], prefix="last_exec_")
            # model.load("./models", name=parameters["base_run_id"])
        # quit()
        # Saving model information:
        print("Saving training information...")
        model.save_model_info("./models", parameters)
        print("Done!")
        train_model(
            model,
            env,
            rb,
            string_log,
            buffer_size_to_train = rb_parameters["minimum_obs_before_training"],
            eval_freq = parameters["eval_frequency"],
            number_of_eval_episodes = parameters["eval_episodes"],
            max_steps = parameters["max_step_count"],
            save_best = True,
            render = False,
            # writer=None
            # writer=SummaryWriter("./models/logs/" + parameters["run_id"]),
            writer=SummaryWriter("./models/logs/" + args.run_id),
            # buffer_op = args.buffer_op,
            curriculum = has_curriculum,
            # use_augmentation = (model.actor.augmentation_capable() and model.critic.augmentation_capable()),
            use_memory = (model.actor.memory_capable() and model.critic.memory_capable()),
            step_update_ratio = parameters["step_update_ratio"],
            augmentation_indexes= aug_parameters["indexes"],
            parameters = parameters
        )
    elif(args.exec_type == "eval"):
        if (args.visualize_input == "true"):
            image = np.zeros((256, 256))
            cv2.imshow('Agent image',image)            
            # cv2.moveWindow('Agent image',int(960-368/2),0)
            # cv2.waitKey(0)
        rec_arch = False         
        if(args.manual_control == "true"):
            model = HumanOperator("./src/Utils/xbox.yaml", env.action_dim)
        elif(args.naive_policy == "true"):
            model = NaiveModel()
            (mr, r_std), (mel, mel_std), (suc, suc_std), ev_steps = eval_model(model, env, parameters["eval_episodes"], rec_arch=False, verbose=True, parameters=parameters, render = (args.visualize_input == "true"))            
        else:
            model_type_str = "best" if args.eval_best == "true" else "latest"
            print("Evaluating model. The %s model will be loaded..." % (model_type_str))
            if(args.eval_best == "true"):
                model.load("./models", prefix="")
            else:
                model.load("./models", prefix="last_exec_")
            rec_arch = (model.actor.memory_capable() and model.critic.memory_capable())
        (mr, r_std), (mel, mel_std), (suc, suc_std), ev_steps = eval_model(model, env, parameters["eval_episodes"], rec_arch=rec_arch, render = (args.visualize_input == "true"), verbose=True, parameters=parameters)
        print("Evaluated the model for %d episodes. Summary:" % (parameters["eval_episodes"]))
        print("\tMean reward %f (± %f)" % (mr, r_std))
        print("\tMean success %.2f%% (± %f%%)" % (suc*100, suc_std*100))
        print("\tMean episode length %f (± %f)" % (mel, mel_std))
        print("\tTotal steps %f" % (ev_steps))
        if(args.manual_control == "true"):
            model.controller.stop()
 
if __name__ == '__main__':
    main()