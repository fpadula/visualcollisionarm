from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import numpy as np

# Sample policy that performs random actions, ignoring the state.
def sample_policy(_):
    # We select a random number between 0 and 1 and rescale between -1 and 1
    return np.random.random_sample((6,1))*2.0 - 1.0

def main():
    # Simulation path:
    env_location = "./simu_envs/SingleAgentVisualization/scene.x86_64"
    # Loading the unity environment:
    unity_env = UnityEnvironment(env_location)
    # Wrapping with the Gym Wrapper:
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    
    # We reset the environment and get our initial state:
    state = env.reset()
    while True:
        # We select an action based on a policy and state:
        action = sample_policy(state)
        # We perform an action in the environment, receiving
        # the next state, the reward and a flag indicating
        # if the episode has ended:
        next_state, reward, ended, _ = env.step(action)
        # If the episode ended, we reset the environment,
        # otherwise we continue execution:
        if ended:
            state = env.reset()
        else:
            state = next_state

if __name__ == '__main__':
    main()