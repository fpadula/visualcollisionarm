import numpy as np
import torch
import os

# Implements a replay buffer with memory
class ReplayBufferM:
    def __init__(self, state_space_dim, action_dim, a_lstm_hidden_dim, c_lstm_hidden_dim, no_of_agents=10, memory_length = 16, buffer_capacity=25000, batch_size=64, device="cpu"):
        self.device = torch.device(device)
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        self.total_valid_samples = 0
        # Num of tuples to train on.
        self.batch_size = batch_size
        # Memory sequence size
        self.memory_length = memory_length
        # If this is a multi observation environment (images and real value obs. for example),
        # 'num_states' represents a list with size bigger than 1. Real valued obs are represented by
        # integers, while images are represented by tuples. Eg.:
        # [4, (1, 50, 50)] represents an environment with real observations with size 4 and images with
        # size 50x50 and one color channel.
        self.state_space_dim = state_space_dim
        self.action_dim = action_dim
        # Size of hidden dimension
        self.a_lstm_hidden_dim = a_lstm_hidden_dim
        self.c_lstm_hidden_dim = c_lstm_hidden_dim
        # print(hidden_dim)
        # quit()
        # Number of agents
        self.no_of_agents = no_of_agents
        # Agent's buffer capacity
        self.agents_capacity = int( (self.buffer_capacity/self.no_of_agents)/ self.memory_length)
        if(self.agents_capacity * self.no_of_agents * self.memory_length != self.buffer_capacity):
            print("Warning: Consider using a buffer size that is multiple of the number of agents in simulation and memory length. Setting buffer size from %d to %d." % (self.buffer_capacity, self.agents_capacity * self.no_of_agents* self.memory_length))
            self.buffer_capacity = self.agents_capacity * self.no_of_agents * self.memory_length


        # buffer_counter stores the last valid memory position
        self.valid_obs_ptr = np.zeros(self.no_of_agents, dtype=np.int16)
        # seq_index_ptr stores the next memory position to be written
        self.seq_index_ptr = np.zeros(self.no_of_agents, dtype=np.int16)
        # pos_in_seq_ptr stores the next sequence position to be written
        self.pos_in_seq_ptr = np.zeros(self.no_of_agents, dtype=np.int16)

        self.buffer_type = torch.float
        self.req_g = False
        # Store everything in a pytorch tensor
        self.action_buffer = torch.zeros(self.memory_length, self.no_of_agents, self.agents_capacity, self.action_dim, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)
        self.reward_buffer = torch.zeros(self.memory_length, self.no_of_agents, self.agents_capacity, 1, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)
        self.done_buffer = torch.zeros(self.memory_length, self.no_of_agents, self.agents_capacity, 1, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)

        self.a_cell_state_buffer = torch.zeros(self.memory_length, self.no_of_agents, self.agents_capacity, self.a_lstm_hidden_dim, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)
        # We need to add a second dimension to the critic cell state buffer because in TD3 we have 2 critic networks
        self.c_cell_state_buffer = torch.zeros(2, self.memory_length, self.no_of_agents, self.agents_capacity, self.c_lstm_hidden_dim, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)

        # quit()
        self.state_buffer = []
        self.next_state_buffer = []
        for ssd in self.state_space_dim:
            if not isinstance(ssd, tuple):
                ssd = (ssd,)
            self.state_buffer.append(torch.zeros( (self.memory_length, self.no_of_agents, self.agents_capacity,) + ssd, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device))
            self.next_state_buffer.append(torch.zeros( (self.memory_length, self.no_of_agents, self.agents_capacity, ) + ssd, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device))


    def save(self, path = "", filename="buffer.pt"):
        if (path != "") and (path[-1] != '/'):
            path = path + '/'
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        torch.save(self, path + filename)

    def reset_seq_ptr(self):
        self.pos_in_seq_ptr = np.zeros(self.no_of_agents, dtype=np.int16)

    @staticmethod
    def load(path = "", filename="buffer.pt", device = "cpu"):
        if (path != "") and (path[-1] != '/'):
            path = path + '/'
        return torch.load(path + filename, map_location=torch.device(device))


    def _process_tuple(self, tuple):
        if not isinstance(tuple, torch.Tensor):
            tuple = torch.tensor(tuple, requires_grad = self.req_g, dtype=self.buffer_type).to(self.device)
        else:
            if tuple.get_device() != self.device:
                tuple = tuple.to(self.device)
        return tuple

    # # Takes (s,a,r,s',done) tuple as input for a group of agents
    # def record(self, prev_states, actions, rewards, states, dones, agent_ids):
    #     for index, id in enumerate(agent_ids):
    #         self.single_record(prev_states, actions, rewards,states,dones, id, index)

    # Takes (s,a,r,s',done) tuple as input for agent with id agent_id
    def record(self, prev_states, actions, rewards, states, dones, a_cell_states, c_cell_states, agent_ids):
    # def record(self, prev_states, actions, rewards, states, dones, agent_ids):
        # Set index to zero if agents_capacity is exceeded,
        # replacing old records
        indexes_to_reset = agent_ids[(self.seq_index_ptr[agent_ids] >= self.agents_capacity)]
        if np.any(indexes_to_reset.size):
            self.seq_index_ptr[indexes_to_reset] = 0
        agents_that_ended = agent_ids[(dones == 1.0).reshape(-1)]
        for agent in agents_that_ended:
            # If the terminal state is not in the last position of the sequence,
            # we need to shift the terminal state to the last position and fill
            # with states from the previous sequence
            if(self.pos_in_seq_ptr[agent] != self.memory_length - 1):
                # How much we will have to shift the observations
                s_amm = self.memory_length - 1 - self.pos_in_seq_ptr[agent]
                s_b_index = self.seq_index_ptr[agent]

                src_start = 0
                src_end = self.pos_in_seq_ptr[agent]
                dest_start = s_amm
                dest_end = self.memory_length - 1
                # Shifting observations inside sequence:
                self.action_buffer[dest_start:dest_end, agent, s_b_index] = self.action_buffer[src_start:src_end, agent, s_b_index].clone()
                self.reward_buffer[dest_start:dest_end, agent, s_b_index] = self.reward_buffer[src_start:src_end, agent, s_b_index].clone()
                self.done_buffer[dest_start:dest_end, agent, s_b_index] = self.done_buffer[src_start:src_end, agent, s_b_index].clone()
                for i, ob_type in enumerate(zip(prev_states, states)):
                    self.state_buffer[i][dest_start:dest_end, agent, s_b_index] = self.state_buffer[i][src_start:src_end, agent, s_b_index].clone()
                    self.next_state_buffer[i][dest_start:dest_end, agent, s_b_index] = self.next_state_buffer[i][src_start:src_end, agent, s_b_index].clone()
                self.a_cell_state_buffer[dest_start:dest_end, agent, s_b_index] = self.a_cell_state_buffer[src_start:src_end, agent, s_b_index].clone()
                self.c_cell_state_buffer[:, dest_start:dest_end, agent, s_b_index] = self.c_cell_state_buffer[:, src_start:src_end, agent, s_b_index].clone()

                # Now we need to populate the remaining sequence elements with observations from previous sequence
                # We need to get 's_amm' observations from previous sequence
                ps_b_index = s_b_index - 1
                if(ps_b_index < 0):
                    # We need to get sequences from the last sequence batch index possible:
                    ps_b_index = self.agents_capacity - 1
                    if(ps_b_index > self.valid_obs_ptr[agent] - 1):
                        # This means that we do not have a valid sequence to pull observations from.
                        # This happens when the agents finish its first sequence with less than
                        # memory_length steps
                        print("Error. At least one sequence to pull observations from is needed. Aborting...")
                        print(self.valid_obs_ptr[agent] - 1)
                        print(ps_b_index)
                        quit()

                src_start = self.memory_length - s_amm
                src_end = self.memory_length
                dest_start = 0
                dest_end = s_amm

                self.action_buffer[dest_start:dest_end, agent, s_b_index] = self.action_buffer[src_start:src_end, agent, ps_b_index].clone()
                self.reward_buffer[dest_start:dest_end, agent, s_b_index] = self.reward_buffer[src_start:src_end, agent, ps_b_index].clone()
                self.done_buffer[dest_start:dest_end, agent, s_b_index] = self.done_buffer[src_start:src_end, agent, ps_b_index].clone()
                for i, ob_type in enumerate(zip(prev_states, states)):
                    self.state_buffer[i][dest_start:dest_end, agent, s_b_index] = self.state_buffer[i][src_start:src_end, agent, ps_b_index].clone()
                    self.next_state_buffer[i][dest_start:dest_end, agent, s_b_index] = self.next_state_buffer[i][src_start:src_end, agent, ps_b_index].clone()
                self.a_cell_state_buffer[dest_start:dest_end, agent, s_b_index] = self.a_cell_state_buffer[src_start:src_end, agent, ps_b_index].clone()
                self.c_cell_state_buffer[:, dest_start:dest_end, agent, s_b_index] = self.c_cell_state_buffer[:, src_start:src_end, agent, ps_b_index].clone()
                self.pos_in_seq_ptr[agent] = self.memory_length - 1
        # Getting index of where the sequence is going to be written
        b_index = self.seq_index_ptr[agent_ids]
        # Getting index of where the observation of the sequence is going to be written
        m_index = self.pos_in_seq_ptr[agent_ids]

        # print(agents_that_ended)

        self.action_buffer[m_index, agent_ids, b_index] = self._process_tuple(actions)
        self.reward_buffer[m_index, agent_ids, b_index] = self._process_tuple(rewards)
        self.done_buffer[m_index, agent_ids, b_index] = self._process_tuple(dones)
        for i, ob_type in enumerate(zip(prev_states, states)):
            self.state_buffer[i][m_index, agent_ids, b_index] = self._process_tuple(ob_type[0])
            self.next_state_buffer[i][m_index, agent_ids, b_index] = self._process_tuple(ob_type[1])
        self.a_cell_state_buffer[m_index, agent_ids, b_index] = self._process_tuple(a_cell_states)
        self.c_cell_state_buffer[:, m_index, agent_ids, b_index] = self._process_tuple(c_cell_states)
        self.pos_in_seq_ptr[agent_ids] += 1
        indexes_to_reset_seq_ptr = (self.pos_in_seq_ptr >= self.memory_length)
        if np.any(indexes_to_reset_seq_ptr):
            agent_ids = np.arange(self.no_of_agents)
            self.pos_in_seq_ptr[agent_ids[indexes_to_reset_seq_ptr]] = 0
            self.seq_index_ptr[agent_ids[indexes_to_reset_seq_ptr]] += 1
            self.valid_obs_ptr[agent_ids[indexes_to_reset_seq_ptr]] = np.minimum(self.valid_obs_ptr[agent_ids[indexes_to_reset_seq_ptr]]+1, self.agents_capacity)
            self.total_valid_samples = np.sum(self.valid_obs_ptr)

    # Samples a batch of (s,a,r,s') observation tuple
    def sample(self, to_device = None):
        # self.no_of_agents, self.agents_capacity, self.memory_length


        sample_indexes = np.random.choice(self.total_valid_samples, size = self.batch_size, replace=(self.total_valid_samples < self.batch_size))
        agents_indexes = np.zeros(self.batch_size, dtype=np.int16)
        batch_indexes = np.zeros(self.batch_size, dtype=np.int16)

        # print("###################################")
        # print(self.total_valid_samples)
        # print(sample_indexes)
        # print(self.valid_obs_ptr)
        for i in range(self.batch_size):
            j = 0
            can_index_batch = False
            while(not can_index_batch):
                if(sample_indexes[i] < self.valid_obs_ptr[j]):
                    can_index_batch = True
                else:
                    sample_indexes[i]-= self.valid_obs_ptr[j]
                    agents_indexes[i] += 1
                    j+=1
            if(self.seq_index_ptr[agents_indexes[i]] == sample_indexes[i]):
                # This sequence is not yet filled, so we can't use it.
                # We should get the previous sequence, as it is newer than the next
                sample_indexes[i]-=1
                if(sample_indexes[i] < 0):
                    sample_indexes[i] = self.agents_capacity - 1
            batch_indexes[i] = sample_indexes[i]
        # print(sample_indexes)
        # print(agents_indexes)
        # print(batch_indexes)
        # print("###################################")
        # # Get sampling range
        # # record_range = np.minimum(self.valid_obs_ptr, self.buffer_capacity)
        # record_range = self.valid_obs_ptr
        # # Randomly sample indexes
        # agents_indexes = np.random.choice(self.no_of_agents, self.batch_size)
        # batch_indexes = np.zeros(self.batch_size, dtype=np.int16)

        # for i in range(self.batch_size):
        #     # Remove self.record_index[agent_id] from record_range
        #     possible_indexes = np.arange(record_range[agents_indexes[i]])
        #     if (self.seq_index_ptr[agents_indexes[i]] < record_range[agents_indexes[i]]):
        #         possible_indexes = np.delete(possible_indexes, self.seq_index_ptr[agents_indexes[i]])
        #     batch_indexes[i] = np.random.choice(possible_indexes)

        action_batch = self.action_buffer[:, agents_indexes, batch_indexes]
        reward_batch = self.reward_buffer[:, agents_indexes, batch_indexes]
        done_batch = self.done_buffer[:, agents_indexes, batch_indexes]
        # Gets the first cell state of the sequence:
        # quit()
        # print(self.c_cell_state_buffer[0].shape)
        # print(self.c_cell_state_buffer[agents_indexes, :, batch_indexes,0].shape)
        actor_cell_states = self.a_cell_state_buffer[0, agents_indexes, batch_indexes].unsqueeze(dim=0)
        critic_cell_states = self.c_cell_state_buffer[:, 0, agents_indexes, batch_indexes]
        # actor_cell_states = self.a_cell_state_buffer[agents_indexes, batch_indexes, 0].unsqueeze(dim=0)
        # critic_cell_states = self.c_cell_state_buffer[:, agents_indexes, batch_indexes, 0]
        if(to_device != None):
            action_batch = action_batch.to(to_device)
            reward_batch = reward_batch.to(to_device)
            done_batch = done_batch.to(to_device)
            actor_cell_states = actor_cell_states.to(to_device)
            critic_cell_states = critic_cell_states.to(to_device)
        state_batch = []
        next_state_batch = []
        for sample in zip(self.state_buffer, self.next_state_buffer):
            state_batch_obs = sample[0][:, agents_indexes, batch_indexes]
            next_state_batch_obs = sample[1][:, agents_indexes, batch_indexes]
            if(to_device != None):
                state_batch_obs = state_batch_obs.to(to_device)
                next_state_batch_obs = next_state_batch_obs.to(to_device)
            state_batch.append(state_batch_obs)
            next_state_batch.append(next_state_batch_obs)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, actor_cell_states, critic_cell_states)

    def print(self, batch=None, agent_ids = None, print_shapes=False):
        if batch is None:
            agent_ids = np.arange(self.no_of_agents)
        for a_i, id in enumerate(agent_ids):
            if batch is None:
                sb = self.state_buffer
                nsb = self.next_state_buffer
                ab = self.action_buffer[:, id]
                rb = self.reward_buffer[:, id]
                db = self.done_buffer[:, id]
                ahb = self.a_cell_state_buffer[:, id]
                chb = self.c_cell_state_buffer[:,:, id]
                num_obs = self.valid_obs_ptr[id]
                print("Agent ID (%d):" % id)
                for k in range(num_obs):
                    print('\tSequence (%d):' % (k))
                    for i in range(self.memory_length):
                        print('\t\t(%d): (s,a,r,s\',done) = (' % (i), end="")
                        for state_ob in sb:
                            print(state_ob[i, id, k].data.numpy().round(2), end = ",")
                        print(")", end = ",")
                        print(ab[i,k].data.numpy().round(2), end = ",")
                        print(rb[i,k].data.numpy().round(2), end = ",")
                        print("(", end = "")
                        for nstate_ob in nsb:
                            print(nstate_ob[i, id, k].numpy().round(2), end = ",")
                        print(")", end = ",")
                        print(db[i, k].data.numpy().round(2), end = " # ")
                        print(ahb[i, k].data.numpy().round(2), end = ",")
                        print(chb[:, i, k].data.numpy().round(2))
            else:
                sb = batch[0]
                nsb = batch[3]
                ab = batch[1][a_i]
                rb = batch[2][a_i]
                db = batch[4][a_i]
                num_obs = 1
                print("Experience at position (%d). Agent id (%d):" % (a_i, id))
                # print('\tSequence number (%d):' % (k))
                for i in range(self.memory_length):
                    print('\t(%d): (s,a,r,s\',done) = (' % (i), end="")
                    for state_ob in sb:
                        print(state_ob[a_i][i].data.numpy().round(2), end = ",")
                    print(")", end = ",")
                    print(ab[i].data.numpy().round(2), end = ",")
                    print(rb[i].data.numpy().round(2), end = ",")
                    print("(", end = "")
                    for nstate_ob in nsb:
                        print(nstate_ob[a_i][i].numpy().round(2), end = ",")
                    print(")", end = ",")
                    print(db[i].data.numpy().round(2))

def main():
    # Testing class 'ReplayBuffer'

    # # Single obs testing:
    # rb = ReplayBuffer(state_space_dim=[3], action_dim=2, buffer_capacity=10, batch_size=2)
    # # Generating observations and storing on the replay buffer
    # for i in range(10):
    #     obs = ([torch.rand(3)], torch.rand(2), torch.rand(1), [torch.rand(3)], torch.tensor(1))
    #     rb.record(obs)
    # # Printing whole buffer
    # rb.print(print_shapes=False)
    # # quit()
    # # Sampling and printing
    # sample = rb.sample()
    # rb.print(sample)
    # # Adding 5 more samples to the buffer
    # print(rb.buffer_counter)
    # for i in range(5):
    #     obs = ([torch.rand(3)], torch.rand(2), torch.rand(1), [torch.rand(3)], torch.tensor(1))
    #     rb.record(obs)
    # # Printing whole buffer
    # rb.print()
    torch.manual_seed(0)
    # Multi-obs testing
    # rb = ReplayBufferM(state_space_dim=[8,(1, 2, 2)], action_dim=2, buffer_capacity=10, batch_size=2)
    rb = ReplayBufferM(state_space_dim=[(8,)], action_dim=2, memory_length=4, no_of_agents=5, buffer_capacity=10, batch_size=3)
    # rb = ReplayBuffer.load()
    # Generating observations and storing on the replay buffer
    # for i in range(10):
    #     # (s,a,r,s')
    #     rb.record([torch.rand(1,8), torch.zeros(1,1,2,2)], torch.rand(1,2), torch.rand(1,1), [torch.rand(1,8), torch.zeros(1,1,2,2)], torch.tensor(1).reshape(1,1))
    for _ in range(4):
        rb.record([torch.rand(5, 8)], torch.rand(5, 2), torch.rand(5,1), [torch.rand(5, 8)], torch.ones(5,1), agent_ids=np.arange(5))
    for _ in range(4):
        rb.record([torch.rand(5, 8)], torch.rand(5, 2), torch.rand(5,1), [torch.rand(5, 8)], torch.zeros(5,1), agent_ids=np.arange(5))
    for _ in range(4):
        rb.record([torch.rand(5, 8)], torch.rand(5, 2), torch.rand(5,1), [torch.rand(5, 8)], torch.ones(5,1)*5, agent_ids=np.arange(5))

    # Printing whole buffer
    # rb.print()
    # quit()
    # Sampling and printing
    # (state_batch, action_batch, reward_batch, next_state_batch, done_batch) = rb.sample()
    sample, ids = rb.sample()
    # print(sample)
    rb.print()
    rb.print(sample, ids)
    quit()
    # rb.print(sample)
    # # print(rb.state_buffer[0][[1, 2, 5]])
    # # exit()
    # # Adding 2 more samples to the buffer
    # for i in range(2):
    #     obs = ([torch.rand(8), torch.zeros(1,2,2)], torch.rand(2), torch.rand(1), [torch.rand(8), torch.zeros(1,2,2)], torch.tensor(1))
    #     rb.record(obs)
    # # Printing whole buffer
    # rb.print()
    # # rb.save()

if __name__ == "__main__":
    main()