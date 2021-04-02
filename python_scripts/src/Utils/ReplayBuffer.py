import numpy as np
import torch
import os

class ReplayBuffer:
    def __init__(self, state_space_dim, action_dim, buffer_capacity=100000, batch_size=64, device="cpu"):
        self.device = torch.device(device)
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # buffer_counter stores the last valid memory position, and record_index the next position to record
        self.buffer_counter = 0
        self.record_index = 0

        # If this is a multi observation environment (images and real value obs. for example),
        # 'num_states' represents a list with size bigger than 1. Real valued obs are represented by
        # integers, while images are represented by tuples. Eg.:
        # [4, (1, 50, 50)] represents an environment with real observations with size 4 and images with
        # size 50x50 and one color channel.        
        self.state_space_dim = state_space_dim        
        self.action_dim = action_dim

        self.buffer_type = torch.float
        self.req_g = False
        # Store everything in a pytorch tensor
        self.action_buffer = torch.zeros(self.buffer_capacity, self.action_dim, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)
        self.reward_buffer = torch.zeros(self.buffer_capacity, 1, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)
        self.done_buffer = torch.zeros(self.buffer_capacity, 1, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device)
        
        self.state_buffer = []
        self.next_state_buffer = []        
        for ssd in self.state_space_dim:
            if not isinstance(ssd, tuple):
                ssd = (ssd,)
            self.state_buffer.append(torch.zeros( (self.buffer_capacity,) + ssd, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device))
            self.next_state_buffer.append(torch.zeros( (self.buffer_capacity,) + ssd, dtype=self.buffer_type, requires_grad = self.req_g).to(self.device))

    def save(self, path = "", filename="buffer.pt"):
        if (path != "") and (path[-1] != '/'):
            path = path + '/'
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        torch.save(self, path + filename)

    @staticmethod
    def load(path = "", filename="buffer.pt", device = "cpu"):
        if (path != "") and (path[-1] != '/'):
            path = path + '/'
        return torch.load(path + filename, map_location=torch.device(device))


    def _process_tuple(self, tuple):
        if not isinstance(tuple, torch.Tensor):        
            tuple = torch.tensor(tuple, requires_grad = self.req_g, dtype=self.buffer_type).to(self.device)                            
        return tuple
    
    # Takes (s,a,r,s',done) obervation tuple as input
    def record(self, prev_states, actions, rewards, states, dones, agent_ids):
        if (actions.shape[0] == rewards.shape[0]) and (rewards.shape[0] == dones.shape[0]):
            batch_size = rewards.shape[0]
        else:
            print("Failed recording tuple, aborting...")
            quit()
        for prev_obs_type, obs_type in zip(prev_states, states):
            if (prev_obs_type.shape[0] != batch_size) or (obs_type.shape[0] != batch_size):
                print("Failed recording tuple, aborting...")
                quit()

        buffer_overflow = False
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        if(self.record_index >= self.buffer_capacity):
            self.record_index = 0        
        
        s_index = self.record_index
        e_index = s_index + batch_size
        record_size = batch_size
        if (e_index > self.buffer_capacity):
            buffer_overflow = True
            e_index = self.buffer_capacity
            record_size = e_index - s_index

        self.action_buffer[s_index:e_index] = self._process_tuple(actions[:record_size])
        self.reward_buffer[s_index:e_index] = self._process_tuple(rewards[:record_size])
        self.done_buffer[s_index:e_index] = self._process_tuple(dones[:record_size])

        for i, ob_type in enumerate(zip(prev_states, states)):                
            self.state_buffer[i][s_index:e_index] = self._process_tuple(ob_type[0][:record_size])
            self.next_state_buffer[i][s_index:e_index] = self._process_tuple(ob_type[1][:record_size])        

        self.record_index += record_size
        self.buffer_counter = min(self.buffer_counter+record_size, self.buffer_capacity)

        # Records the rest of the observations
        if(buffer_overflow):
            # For now just ignore...
            pass

    # Samples a batch of (s,a,r,s', ids) observation tuple
    def sample(self, to_device = None):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        done_batch = self.done_buffer[batch_indices]
        if(to_device != None):
            action_batch = action_batch.to(to_device)
            reward_batch = reward_batch.to(to_device)
            done_batch = done_batch.to(to_device)
        state_batch = []
        next_state_batch = []
        for sample in zip(self.state_buffer, self.next_state_buffer):
            state_batch_obs = sample[0][batch_indices]
            next_state_batch_obs = sample[1][batch_indices]
            if(to_device != None):
                state_batch_obs = state_batch_obs.to(to_device)
                next_state_batch_obs = next_state_batch_obs.to(to_device)
            state_batch.append(state_batch_obs)
            next_state_batch.append(next_state_batch_obs)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def print(self, batch=None, print_shapes=False):
        if batch is None:
            num_obs = self.buffer_counter
            sb = self.state_buffer
            ab = self.action_buffer
            rb = self.reward_buffer
            nsb = self.next_state_buffer
            db = self.done_buffer
        else:
            sb = batch[0]
            ab = batch[1]
            rb = batch[2]
            nsb = batch[3]
            db = batch[4]
            num_obs = db.shape[0]

        for i in range(num_obs):
            if(print_shapes):
                print('%d: (s,a,r,s`,done) = ' % (i), end="")                
                print("(", end = "")
                for state_ob in sb:
                    print(state_ob[i].shape, end = ",")
                print(")", end = ",")                
                print(ab[i].data.shape, end = ",")
                print(rb[i].data.shape, end = ",")                
                print("(", end = "")
                for nstate_ob in nsb:
                    print(nstate_ob[i].shape, end = ",")
                print(")", end = ",")                
                print(db[i].data.shape)
            else:
                print('%d: (s,a,r,s`,done) = ' % (i), end="")                
                print("(", end = "")
                for state_ob in sb:
                    print(state_ob[i].data.numpy().round(2), end = ",")
                print(")", end = ",")                          
                print(ab[i].data.numpy().round(2), end = ",")
                print(rb[i].data.numpy().round(2), end = ",")                
                print("(", end = "")
                for nstate_ob in nsb:
                    print(nstate_ob[i].numpy().round(2), end = ",")
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
    rb = ReplayBuffer(state_space_dim=[8,(1, 2, 2)], action_dim=2, buffer_capacity=10, batch_size=2)
    # rb = ReplayBuffer.load()
    # Generating observations and storing on the replay buffer    
    # for i in range(10):
    #     # (s,a,r,s')        
    #     rb.record([torch.rand(1,8), torch.zeros(1,1,2,2)], torch.rand(1,2), torch.rand(1,1), [torch.rand(1,8), torch.zeros(1,1,2,2)], torch.tensor(1).reshape(1,1))
    rb.record([torch.rand(10,8), torch.zeros(10,1,2,2)], torch.rand(10,2), torch.rand(10,1), [torch.rand(10,8), torch.zeros(10,1,2,2)], torch.ones(10,1))
    # Printing whole buffer
    rb.print()
    rb.record([torch.rand(10,8), torch.zeros(10,1,2,2)], torch.rand(10,2), torch.rand(10,1), [torch.rand(10,8), torch.zeros(10,1,2,2)], torch.ones(10,1))
    rb.print()
    # Sampling and printing
    # (state_batch, action_batch, reward_batch, next_state_batch, done_batch) = rb.sample()
    sample = rb.sample()
    # # print(state_batch[1].shape)
    print(sample)
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