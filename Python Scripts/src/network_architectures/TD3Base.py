from network_architectures.Base import BaseActor, BaseCritic
import torch
import torch.nn as nn
# from Initializations import layer_init, hidden_layer_init

class TD3BaseActor(BaseActor):
    def __init__(self, state_dims, action_dims, output_scale=1.0):
        super(TD3BaseActor, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims        
        self.output_scale = output_scale               

        self.l1 = nn.Linear(self.state_dims[0][0], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, self.action_dims)

        self.model = nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3,
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x[0])*self.output_scale

class TD3BaseCritic(BaseCritic):
    def __init__(self, state_dims, action_dims):        
        super(TD3BaseCritic, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.augmented_capable = False

        # Q1 architecture
        self.l1 = nn.Linear(self.state_dims[0][0] + self.action_dims, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(self.state_dims[0][0] + self.action_dims, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.model_Q1 = nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3
        )

        self.model_Q2 = nn.Sequential(
            self.l4,
            nn.ReLU(),
            self.l5,
            nn.ReLU(),
            self.l6
        )

    def forward(self, states, actions):
        sa = torch.cat([states[0], actions], 1)
        q1output = self.model_Q1(sa)
        q2output = self.model_Q2(sa)
        return q1output, q2output

    def Q1(self, states, actions):
        return self.model_Q1(torch.cat([states[0], actions], 1))        