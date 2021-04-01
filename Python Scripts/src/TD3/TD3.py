import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import sys
# from network_architectures.TD3BaseActor import TD3BaseActor
# from network_architectures.TD3BaseCritic import TD3BaseCritic
from network_architectures import *
# import network_architectures

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(state_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, action_dim)

#         self.max_action = max_action


#     def forward(self, state):
#         a = F.relu(self.l1(state))
#         a = F.relu(self.l2(a))
#         return self.max_action * torch.tanh(self.l3(a))


# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()

#         # Q1 architecture
#         self.l1 = nn.Linear(state_dim + action_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, 1)

#         # Q2 architecture
#         self.l4 = nn.Linear(state_dim + action_dim, 256)
#         self.l5 = nn.Linear(256, 256)
#         self.l6 = nn.Linear(256, 1)


#     def forward(self, state, action):
#         sa = torch.cat([state, action], 1)

#         q1 = F.relu(self.l1(sa))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)

#         q2 = F.relu(self.l4(sa))
#         q2 = F.relu(self.l5(q2))
#         q2 = self.l6(q2)
#         return q1, q2


#     def Q1(self, state, action):
#         sa = torch.cat([state, action], 1)

#         q1 = F.relu(self.l1(sa))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)
#         return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        model_name,
        net_config_name,
        device,
        discount=0.99,
        tau=0.005,
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        policy_noise=0.2,
        expl_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        mem_parameters = None
    ):
        # print(state_dim)
        # print(action_dim)
        # print(max_action)
        # print(discount)
        # print(tau)
        # print(policy_noise)
        # print(noise_clip)
        # print(policy_freq)
        # quit()
        self.buffer_type = torch.float
        self.req_g = False
        self.device = torch.device(device)
        self.model_name = model_name
        self.net_config_name = net_config_name
        if(mem_parameters != None):
            self.bperi = mem_parameters["burn_in_period"]

        self.state_dim = state_dim
        self.action_dim = action_dim
        net_architecture = net_config_name + "." + net_config_name
        self.actor = eval(net_architecture + "Actor(state_dim, action_dim, max_action).to(self.device)")
        # self.actor = Actor(state_dim, action_dim, net_config_name, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        print("Actor no of parameters:")
        self.actor_total_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        print(self.actor_total_params)

        self.critic = eval(net_architecture + "Critic(state_dim, action_dim).to(self.device)")
        # self.critic = Critic(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        print("Critic no of parameters:")
        self.critic_total_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print(self.critic_total_params)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.policy_freq = policy_freq

        self.total_it = 0

        # Storing losses for metrics
        self.a_loss = 0
        self.c_loss = 0
        self.aug_a_loss = 0
        self.aug_c_loss = 0

    def prepare_tuple(self, t):
        if not isinstance(t, torch.Tensor):
            return torch.tensor(t, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)
        elif t.get_device() != self.device:
            return t.to(self.device)
        return t

    def prepare_obs(self, obs_list):
        tobs = []
        for ob in obs_list:
            to_add = self.prepare_tuple(ob)
            # tobs.append(torch.unsqueeze(to_add, dim=0))
            if self.actor.memory_capable():
                tobs.append(torch.unsqueeze(to_add, dim=0))
            else:
                tobs.append(to_add)
            # print(tobs[0].shape)
        return tobs

    def react_to_obs(self, obs, memory_variables = None, add_noise=False, t_device=None):
        tobs = self.prepare_obs(obs)
        self.actor.eval()
        with torch.no_grad():
            if memory_variables == None:
                action = self.actor(tobs)
            else:
                (actor_hidden_states, actor_cell_states) = memory_variables[0]
                actor_hidden_states = self.prepare_tuple(actor_hidden_states)
                actor_cell_states = self.prepare_tuple(actor_cell_states)
                action, (actor_hidden_states, actor_cell_states) = self.actor(tobs, actor_hidden_states, actor_cell_states)
            if(add_noise):
                action += torch.normal(torch.zeros((action.shape[0],self.action_dim)), torch.ones((action.shape[0],self.action_dim)) * self.max_action * self.expl_noise).to(self.device)
                action = action.clamp(-self.max_action,self.max_action)
            if memory_variables != None:
                (critic_hidden_states, critic_cell_states) = memory_variables[1]
                self.critic.eval()
                critic_hidden_states = self.prepare_tuple(critic_hidden_states)
                critic_cell_states = self.prepare_tuple(critic_cell_states)
                _,_, (critic_cell_states, critic_hidden_states) = self.critic(tobs, action.unsqueeze(dim=0), critic_hidden_states, critic_cell_states)
                self.critic.train()
        self.actor.train()
        if memory_variables == None:
            return action
        if memory_variables != None:
            if(t_device != None):
                actor_hidden_states = actor_hidden_states.to(t_device)
                actor_cell_states = actor_cell_states.to(t_device)
                critic_hidden_states = critic_hidden_states.to(t_device)
                critic_cell_states = critic_cell_states.to(t_device)
            return action, (actor_hidden_states, actor_cell_states), (critic_hidden_states, critic_cell_states)

    def step(self, replay_buffer, augmented_features=None, recurrent_arch = False, non_visual_index=0):
        self.total_it += 1

        # Sample replay buffer
        if not recurrent_arch:
            (state, action, reward, next_state, done) = replay_buffer.sample(to_device = self.device)
        else:
            (state, action, reward, next_state, done, actor_cell_states, critic_cell_states) = replay_buffer.sample(to_device = self.device)
            # Initial hidden states are set to 0
            actor_hidden_states = torch.zeros((1, replay_buffer.batch_size, replay_buffer.a_lstm_hidden_dim)).to(self.device)
            critic_hidden_states = torch.zeros((2, replay_buffer.batch_size, replay_buffer.c_lstm_hidden_dim)).to(self.device)
            # quit()
        if recurrent_arch:
            # First we need to perform the burn-in technique to have proper hidden/cell states
            # state, action, reward, next_state, done
            # Selecting the burning-period indexes:
            batch_size = replay_buffer.batch_size
            
            bp_action = action[:self.bperi]
            action = action[self.bperi:]
            bp_reward = reward[:self.bperi]
            reward = reward[self.bperi:]
            bp_done = done[:self.bperi]
            done = done[self.bperi:]
            
            seq_length = action.shape[0]
            bp_seq_length = bp_action.shape[0]

            bp_state, bp_next_state = [],[]
            l_state, l_next_state = [],[]
            for ob_type in zip(state, next_state):
                bp_state.append(ob_type[0][:self.bperi])
                bp_next_state.append(ob_type[1][:self.bperi])
                l_state.append(ob_type[0][self.bperi:])
                l_next_state.append(ob_type[1][self.bperi:])
            state = l_state
            next_state = l_next_state


        # If has augmented features, first performs SL update and then RL update
        if(isinstance(augmented_features, np.ndarray)):
            # augmented_features store the index of the observations that will be
            # used for feature augmentation
            # If this does not work well, change for target nets...

            if recurrent_arch:
                # Burn-in:
                # self.actor.eval()
                bp_next_action, (bp_actor_hidden_states, bp_actor_cell_states) = self.actor(bp_next_state, actor_hidden_states, actor_cell_states, faug = False)
                # self.actor.train()
                # Learning sequence
                actor_aug_output, _  = self.actor(next_state, bp_actor_hidden_states, bp_actor_cell_states, faug = True)                
                aug_variables = state[non_visual_index][:,:,augmented_features]                
            else:
                actor_aug_output = self.actor(state, faug = True)
                aug_variables = state[non_visual_index][:,augmented_features]
            actor_faug_loss = F.mse_loss(actor_aug_output, aug_variables)
            self.actor_optimizer.zero_grad()
            actor_faug_loss.backward()
            self.actor_optimizer.step()
            self.aug_a_loss += actor_faug_loss.item()

            if recurrent_arch:
                # Burn-in:
                # self.critic.eval()                
                bp_next_action = bp_next_action.reshape((bp_seq_length,batch_size,) + bp_next_action.shape[1:])
                bp_next_action = bp_next_action.detach()
                _, _, (bp_critic_hidden_states, bp_critic_cell_states) = self.critic(bp_next_state, bp_next_action, critic_hidden_states, critic_cell_states,faug = False)
                # self.critic.train()
                # Learning sequence
                Q1_aug_output, Q2_aug_output, _ = self.critic(state, action, bp_critic_hidden_states, bp_critic_cell_states, faug = True)                
            else:
                Q1_aug_output, Q2_aug_output = self.critic(state, action, faug = True)
            critic_faug_loss = F.mse_loss(Q1_aug_output, aug_variables) + F.mse_loss(Q2_aug_output, aug_variables)
            self.critic_optimizer.zero_grad()
            critic_faug_loss.backward()
            self.critic_optimizer.step()
            self.aug_c_loss += critic_faug_loss.item()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            if recurrent_arch:
                bp_noise = (
                    torch.randn_like(bp_action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                noise = noise.reshape(action.shape)
                # Burn-in:
                # self.actor_target.eval()
                bp_at_action, (bp_actor_hidden_states, bp_actor_cell_states)  = self.actor_target(bp_next_state, actor_hidden_states, actor_cell_states)
                # self.actor_target.train()
                # Learning sequence
                at_action, _  = self.actor_target(next_state, bp_actor_hidden_states, bp_actor_cell_states)
            else:
                at_action = self.actor_target(next_state)
            if recurrent_arch:
                at_action = at_action.reshape((noise.shape[0],noise.shape[1],action.shape[-1]))
                bp_at_action = bp_at_action.reshape((bp_noise.shape[0],bp_noise.shape[1],bp_action.shape[-1]))
            next_action = (at_action + noise).clamp(-self.max_action, self.max_action)
            if recurrent_arch:
                bp_next_action = (bp_at_action + bp_noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            if recurrent_arch:
                # Burn-in:
                # self.critic_target.eval()
                _, _, (bp_critic_hidden_states, bp_critic_cell_states) = self.critic_target(bp_next_state, bp_next_action, critic_hidden_states, critic_cell_states)
                # self.critic_target.train()
                # Learning sequence
                target_Q1, target_Q2, _ = self.critic_target(next_state, next_action, bp_critic_hidden_states, bp_critic_cell_states)
            else:
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.discount * target_Q
            # quit()
        # Get current Q estimates
        if recurrent_arch:
            # Burn-in:
            # self.critic.eval()
            _, _, (bp_critic_hidden_states, bp_critic_cell_states) = self.critic(bp_state, bp_action, critic_hidden_states, critic_cell_states )
            # self.critic.train()
            # Learning sequence
            current_Q1, current_Q2, _ = self.critic(state, action, bp_critic_hidden_states, bp_critic_cell_states )
        else:
            current_Q1, current_Q2 = self.critic(state, action)


        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.c_loss += critic_loss.item()
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            if recurrent_arch:
                # Burn-in:
                # self.actor.eval()
                bp_a_action, (bp_actor_hidden_states, bp_actor_cell_states) = self.actor(bp_state, actor_hidden_states, actor_cell_states)
                # self.actor.train()
                # Learning sequence
                a_action, _ = self.actor(state, bp_actor_hidden_states, bp_actor_cell_states)

                bp_a_action = bp_a_action.reshape((bp_seq_length, batch_size,) + bp_a_action.shape[1:])
                a_action = a_action.reshape((seq_length, batch_size,) + a_action.shape[1:])
            else:
                a_action = self.actor(state)
            if recurrent_arch:
                # a_action = a_action.reshape((noise.shape[0],noise.shape[1],2))
                # Burn-in:
                # self.critic.eval()                
                _, (Q1_lstm_h, Q1_lstm_c) = self.critic.Q1(bp_state, bp_a_action, critic_hidden_states[0].unsqueeze(dim=0), critic_cell_states[0].unsqueeze(dim=0))
                # self.critic.train()                                
                # Learning sequence
                actor_loss, _ = self.critic.Q1(state, a_action, Q1_lstm_h, Q1_lstm_c)
                actor_loss = -actor_loss.mean()
            else:
                actor_loss = -self.critic.Q1(state, a_action).mean()
            ####################################

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.a_loss += actor_loss.item()
            # Update the frozen target models
            self.critic_target.update_weights_from_model(self.critic, tau = self.tau)
            self.actor_target.update_weights_from_model(self.actor, tau = self.tau)

    def save(self, path, prefix=""):
        if(self.model_name == None):
            return
        try:
            os.mkdir(path + "/" + self.model_name + "/")
        except FileExistsError:
            pass
        torch.save(self.critic.state_dict(), path + "/" + self.model_name + "/" + prefix + "critic.pt")
        torch.save(self.critic_optimizer.state_dict(), path + "/" + self.model_name + "/" + prefix + "critic_opt.pt")

        torch.save(self.actor.state_dict(), path + "/" + self.model_name + "/" + prefix + "actor.pt")
        torch.save(self.actor_optimizer.state_dict(), path + "/" + self.model_name + "/" + prefix + "actor_opt.pt")


    def load(self, path, name="", prefix=""):
        if(name == ""):
            name = self.model_name
        self.critic.load_state_dict(torch.load(path + "/" + name + "/" + prefix +  "critic.pt", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(path + "/" + name + "/" + prefix + "critic_opt.pt", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(path + "/" + name + "/" + prefix + "actor.pt", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(path + "/" + name + "/" + prefix + "actor_opt.pt", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

    def save_model_info(self, path, hyperparameters):
        if(path[-1] != '/'):
            path = path + '/'
        path = path + self.model_name + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + "parameters.yaml", 'w') as outfile:
            yaml.dump(hyperparameters, outfile, default_flow_style=False)
        original_stdout = sys.stdout
        with open(path + "actor_info.txt", 'w') as f:
            sys.stdout = f
            print(self.actor)
        with open(path + "critic_info.txt", 'w') as f:
            sys.stdout = f
            print(self.critic)
        sys.stdout = original_stdout
