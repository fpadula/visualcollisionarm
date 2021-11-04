import torch

import itertools
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from mlagents_envs.base_env import BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util
import torch
from PIL import Image
from inputs import get_gamepad
import threading
import time
import yaml
import cv2
import math

def ids2indexes(agent_ids, selected_agent_ids):
    index = np.argsort(agent_ids)
    sorted_index = np.searchsorted(agent_ids[index], selected_agent_ids)
    return np.take(index, sorted_index, mode="clip")

def update_obs_list(obs_list, new_obs, ids):
    for i in range(len(new_obs)):
        obs_list[i][ids] = new_obs[i]

def slice_obs_list(obs_list, ids):
    ret = []
    for i in range(len(obs_list)):
        ret.append(obs_list[i][ids])
    return ret

class MultiAgentUnityEnv():
    def __init__(self, unity_env, action_magnitude=1, encoder = None):
        self._env = unity_env
        self._env.reset()
        # self.behavior_name = "RollerBall"
        self.encoder = encoder
        # print("Behavior names:")
        for name in self._env.behavior_specs.keys(): # Assume just one behavior...            
            # print(name)
            self.behavior_name = name
        # self.behavior_name = "3DFollower?team=0"
        self.behavior_spec = self._env.behavior_specs[self.behavior_name]
        self.action_dim = self.behavior_spec.action_shape
        self.state_dim = self.behavior_spec.observation_shapes.copy()
        self._env_state_dim = self.state_dim.copy()
        self.visual_obs_indexes = []
        self.non_visual_obs_index = 0
        for i in range(len(self.state_dim)):
            if(len(self.state_dim[i]) > 1):
                self.visual_obs_indexes.append(i)
            else:
                self.non_visual_obs_index = i

        for index in self.visual_obs_indexes:
            if(self.encoder == None):
                self.state_dim[index] = (self.state_dim[index][2], self.state_dim[index][0], self.state_dim[index][1])
            else:
                self.state_dim[index] = (self.encoder.latent_size, )
        self.action_magnitude = action_magnitude
        decision_steps, _ = self._env.get_steps(self.behavior_name)
        # CHANGE TO GET DINAMICALLY LATTER
        self.no_of_agents = 25
        # self._env.reset()
        self.last_steps_ids = decision_steps.agent_id
        self.print_id = 0

    def action_space_sample(self, size):
        actions = self.behavior_spec.create_random_action(size)
        # print("Creating random action for %d agents" % (size))
        # print(actions.shape)
        return actions

    def step(self, actions):
        # Sorting actions acording to the agent's ids
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        # for id in agents_that_req_action:
        #     self._env.set_action_for_agent(self.behavior_name, id, actions[id])
        if(actions.size != 0):   
            self._env.set_actions(self.behavior_name, actions)
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)        

        if(decision_steps.agent_id.shape[0] != 0):
            dstate = decision_steps.obs
            dreward = decision_steps.reward.reshape(-1,1)
            ddone = np.zeros((decision_steps.agent_id.shape[0], 1))
            decision = dstate, dreward, ddone, decision_steps.agent_id
            # print("#####################")
            # print("Agent ID:")
            # print(decision_steps.agent_id)
            # print(dreward)
            # print("#####################")
        else:
            decision = None
        
        if(terminal_steps.agent_id.shape[0] != 0):
            tstate = terminal_steps.obs
            treward = terminal_steps.reward.reshape(-1,1)
            tdone = np.ones((terminal_steps.agent_id.shape[0], 1))
            terminals = tstate, treward, tdone, terminal_steps.agent_id
            # print("#####################")
            # print("Agent with following ID terminated:")
            # print(terminal_steps.agent_id)
            # print(treward)
            # print("Decision IDS:")
            # if(decision != None):
            #     print(decision_steps.agent_id)
            # else:
            #     print("None")        
            # print("#####################")
        else:
            terminals = None
        # print("########")
        # print(decision_steps.agent_id)
        # print(terminal_steps.agent_id)
        # print("########")
        for visual_i in self.visual_obs_indexes:
            if(terminals != None):
                terminals[0][visual_i] = terminals[0][visual_i].reshape((terminals[0][visual_i].shape[0],terminals[0][visual_i].shape[3],terminals[0][visual_i].shape[1],terminals[0][visual_i].shape[2]))
            if(decision != None):
                decision[0][visual_i] = decision[0][visual_i].reshape((decision[0][visual_i].shape[0],decision[0][visual_i].shape[3],decision[0][visual_i].shape[1],decision[0][visual_i].shape[2]))
            # states[index] = states[index].reshape((states[index].shape[0],states[index].shape[3],states[index].shape[1],states[index].shape[2]))
        #     if (self.encoder != None):
        #         with torch.no_grad():
        #             states[index] = self.encoder.get_latent_rep(torch.tensor(states[index], dtype=torch.float, requires_grad = False).to(torch.device("cuda:1")))
        return decision, terminals

    def reset(self):
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.behavior_name)
        states = decision_steps.obs        
        for visual_i in self.visual_obs_indexes:                        
            states[visual_i] = states[visual_i].reshape((states[visual_i].shape[0],states[visual_i].shape[3],states[visual_i].shape[1],states[visual_i].shape[2]))
        agents_action = self.action_space_sample(self.no_of_agents)
        agents_prev_state = self.gen_empty_states()
        # step_prev_states, step_agents = env.reset()
        step_actions = np.array([])
        prev_state_was_terminal = np.repeat(False, self.no_of_agents)
        reset_variables = (agents_action, agents_prev_state, step_actions, prev_state_was_terminal, states, decision_steps.agent_id)
        return reset_variables
        # return states, decision_steps.agent_id


    def close(self):
        self._env.close()

    def gen_empty_states(self):
        ret = []
        for i in range(len(self.state_dim)):
            ret.append(np.zeros((self.no_of_agents,) + self.state_dim[i]))
        return ret


class GymEnvStd():

    def __init__(self, env):
        self.env_orig = env

        self.observation_space = env.observation_space
        self.state_dim = [env.observation_space.shape[0]]

        self.action_space = env.action_space
        self.action_dim = env.action_space.shape[0]
        self.action_magnitude = float(env.action_space.high[0])

    def seed(self, seed):
        pass

    def reset(self):
        return [self.env_orig.reset()]

    def render(self):
        return self.env_orig.render()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        # print(action)
        state, reward, done, _ = self.env_orig.step(action)
        info = {
            "success": False
        }
        return [state], reward, done, info

class UnityEnvStd():
    def __init__(self, env, string_log, action_magnitude = 1):
        self.env_orig = env

        self.observation_space = env.observation_space
        self.state_dim = []
        self.fix_image_channel = -1
        for i,dim in enumerate(env.observation_space):
            if(len(dim.shape) == 1):
                self.state_dim.append(dim.shape[0])
            else:
                self.fix_image_channel = i
                self.state_dim.append((dim.shape[2], dim.shape[0], dim.shape[1]))

        self.action_space = env.action_space
        self.action_dim = env.action_space.shape[0]

        self.action_magnitude = action_magnitude
        self.string_log = string_log

    def seed(self, seed):
        pass

    def reset(self):
        obs = self.env_orig.reset()
        if(self.fix_image_channel != -1):
            obs[self.fix_image_channel] = self.set_chanel_first(obs[self.fix_image_channel])
        return obs

    def render(self):
        return self.env_orig.render()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        state, reward, done, _ = self.env_orig.step(action)
        if(self.fix_image_channel != -1):
            state[self.fix_image_channel] = self.set_chanel_first(state[self.fix_image_channel])
        info = {
            "success": (self.string_log.read_last_msg() == "success")
        }
        return state, reward, done, info

    def set_chanel_first(self, t):
        return t.reshape((-1,t.shape[0],t.shape[1]))

class Xbox360Controller():
    def __init__(self, key_codes) -> None:        
        self.update_keys = True
        self.keys = {}
        for code in key_codes:
            self.keys[code] = 0
        self.t = threading.Thread(target=self.update_key_states)
        self.t.start()
    
    def stop(self):
        self.update_keys = False
        self.t.join()
        # self.t.terminate()
        # self.t.

    def update_key_states(self):
        while(self.update_keys):
            events = get_gamepad()
            for event in events:
                if event.code in self.keys:
                    self.keys[event.code] = event.state
                    # print(event.ev_type, event.code, event.state)

    def read_key_state(self, key_code):
        return self.keys[key_code]

class HumanOperator():

    def __init__(self, filename, no_actions) -> None:
        with open(filename) as file:            
            self.key_mappings = yaml.load(file, Loader=yaml.FullLoader)
        self.no_actions = no_actions
        self.controller = Xbox360Controller(self.key_mappings.keys())        
        
    def map(self, value, from1, to1, from2, to2):
        return value*((to1 - to2)/(from1 - from2)) + (-to1*from2 + from1 * to2)/(from1 - from2)    

    def react_to_obs(self, state, add_noise=False):        
        actions = np.zeros((1, self.no_actions))            
        for key_name in self.key_mappings:
            key = self.key_mappings[key_name]
            raw_value = self.controller.read_key_state(key_name)
            if(abs(raw_value) <= key["deadzone"]):
                raw_value = 0            
            actions[0, key["action_index"]] = self.map(raw_value, key["src_min"], key["dest_min"], key["src_max"], key["dest_max"])
        return actions

class NaiveModel():

    def __init__(self):
        self.step = 1

    def norm(self,vector):
        ret = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
        if(ret == 0):
            ret = 1
        return ret

    def react_to_obs(self, state, add_noise=False): 
        visual_state = state[0][0]
        scalar_state = state[1][0]
        target_pos = scalar_state[3:6]
        current_pos = scalar_state[6:9]
        dir = target_pos - current_pos
        dir = dir/self.norm(dir)
        dir = self.step * dir
        
        action = np.array([0, 0, 0, dir[0], dir[1], dir[2]])
        return action.reshape(1,-1)

        