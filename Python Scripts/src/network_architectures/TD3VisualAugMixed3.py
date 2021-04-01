from network_architectures.Base import BaseActor, BaseCritic

import torch.nn as nn
import torch
# from Initializations import layer_init, hidden_layer_init

# Networks adapted from Lobos-tsunekawa et. al. (2018), but without LSTMs

class TD3VisualAugMixed3Actor(BaseActor):
    def __init__(self, state_dims, action_dims, output_scale=1.0):
        super(TD3VisualAugMixed3Actor, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.output_scale = output_scale
        self.non_aug_parameters_no = 23
        self.aug_parameters_no = 3

        self.conv1 = nn.Conv2d(self.state_dims[0][0], 16, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=1568, out_features=128)

        self.visual_encoder = nn.Sequential(
                # First conv layer Kernel size set to 4 instead of 8
                self.conv1,
                # Using LeakyReLU instead of ReLU as it shows better convergence
                nn.LeakyReLU(),
                self.conv2,
                nn.LeakyReLU(),
                nn.Flatten(),
                self.fc1,
                # nn.LeakyReLU(),
                # nn.Linear(in_features=128, out_features=128)
            )

        self.fc3 = nn.Linear(in_features = self.non_aug_parameters_no + 128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=self.action_dims)

        self.policy = nn.Sequential(
            # 9 = non-aug scalar features
            self.fc3,
            nn.LeakyReLU(),
            self.fc4,
            nn.LeakyReLU(),
            self.fc5,
            nn.Tanh()
        )

        self.fc2 = nn.Linear(in_features=128, out_features=self.aug_parameters_no)

        self.faug = nn.Sequential(
            self.fc2
        )

    def forward(self, x, faug=False):
        encoded_input = self.visual_encoder(x[0])
        sinput = torch.cat([encoded_input, x[1][:,self.aug_parameters_no:] ], 1)
        if(faug):
            return self.faug(encoded_input)
        else:
            return self.policy(sinput)*self.output_scale

    def augmentation_capable(self):
        return True

class TD3VisualAugMixed3Critic(BaseCritic):
    def __init__(self, state_dims, action_dims):
        super(TD3VisualAugMixed3Critic, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.non_aug_parameters_no = 23
        self.aug_parameters_no = 3

        # Q1 architecture

        self.Q1_conv1 = nn.Conv2d(self.state_dims[0][0], 16, kernel_size=4, stride=4)
        self.Q1_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.Q1_fc1 = nn.Linear(in_features=1568, out_features=128)

        self.Q1_encoding = nn.Sequential(
            self.Q1_conv1,
            nn.LeakyReLU(),
            self.Q1_conv2,
            nn.LeakyReLU(),
            nn.Flatten(),
            self.Q1_fc1,
            # nn.LeakyReLU(),
            # nn.Linear(128, self.aug_parameters_no)
        )

        self.Q1_fc3 = nn.Linear(128 + self.action_dims + self.non_aug_parameters_no, 256)
        self.Q1_fc4 = nn.Linear(256, 256)
        self.Q1_fc5 = nn.Linear(256, 1)

        self.Q1_linear = nn.Sequential(
            self.Q1_fc3,
            nn.LeakyReLU(),
            self.Q1_fc4,
            nn.LeakyReLU(),
            self.Q1_fc5
        )

        self.Q1_fc2 = nn.Linear(in_features=128, out_features=self.aug_parameters_no)

        self.Q1_aug = nn.Sequential(
            self.Q1_fc2
        )

        # Q2 architecture
        self.Q2_conv1 = nn.Conv2d(self.state_dims[0][0], 16, kernel_size=4, stride=4)
        self.Q2_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.Q2_fc1 = nn.Linear(in_features=1568, out_features=128)

        self.Q2_encoding = nn.Sequential(
            self.Q2_conv1,
            nn.LeakyReLU(),
            self.Q2_conv2,
            nn.LeakyReLU(),
            nn.Flatten(),
            self.Q2_fc1,
            # nn.LeakyReLU(),
            # nn.Linear(128, self.aug_parameters_no)
        )

        self.Q2_fc3 = nn.Linear(128 + self.action_dims + self.non_aug_parameters_no, 256)
        self.Q2_fc4 = nn.Linear(256, 256)
        self.Q2_fc5 = nn.Linear(256, 1)

        self.Q2_linear = nn.Sequential(
            self.Q2_fc3,
            nn.LeakyReLU(),
            self.Q2_fc4,
            nn.LeakyReLU(),
            self.Q2_fc5
        )

        self.Q2_fc2 = nn.Linear(in_features=128, out_features=self.aug_parameters_no)

        self.Q2_aug = nn.Sequential(
            self.Q2_fc2
        )

    def forward(self, states, actions, faug = False):
        Q1_encoding = self.Q1_encoding(states[0])
        Q2_encoding = self.Q2_encoding(states[0])

        sa_Q1 = torch.cat([Q1_encoding, actions, states[1][:,self.aug_parameters_no:]], 1)
        sa_Q2 = torch.cat([Q2_encoding, actions, states[1][:,self.aug_parameters_no:]], 1)
        if(faug):
            return self.Q1_aug(Q1_encoding), self.Q2_aug(Q2_encoding)
        else:
            return self.Q1_linear(sa_Q1), self.Q2_linear(sa_Q2)

    def Q1(self, states, actions):
        visual_Q1_out = self.Q1_encoding(states[0])
        return self.Q1_linear(torch.cat([visual_Q1_out, actions, states[1][:,self.aug_parameters_no:]], 1))

    def augmentation_capable(self):
        return True