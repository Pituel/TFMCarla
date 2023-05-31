import os
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR, MODE

import numpy as np
import torch.autograd as autograd


class DuelingDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(DuelingDQnetwork, self).__init__()
        if MODE == 1:
            self.input_shape = (95 + 5,)
        else:
            self.input_shape = (3, 160, 80)
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR, model)
        self.action_space = np.arange(self.n_actions)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        if MODE == 1:
            self.Linear1 = nn.Sequential(
                nn.Linear(95 + 5, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )

        else:
        # Conv
            self.Linear1 = nn.Sequential(
            nn.Conv2d(3 , 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            )

        if torch.cuda.is_available():
            self.Linear1.cuda()
        
        self.fc_layer_inputs = self.feature_size()

        if MODE == 1:
            self.V = nn.Sequential(
                nn.Linear(self.fc_layer_inputs, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.A = nn.Sequential(
                nn.Linear(self.fc_layer_inputs, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_actions)
            )
        else:
            self.V = nn.Sequential(
                nn.Linear(self.fc_layer_inputs, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.A = nn.Sequential(
                nn.Linear(self.fc_layer_inputs, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_actions)
            )

        if self.device == 'cuda':
            self.V.cuda()
            self.A.cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=DQN_LEARNING_RATE)


    def forward(self, x):
        if MODE == 1:
            fc = self.Linear1(x)
        else:
            try:
                fc = self.Linear1(x).reshape(-1, self.fc_layer_inputs)
            except Exception as ex:
                print(ex)
        V = self.V(fc)
        A = self.A(fc)

        if MODE == 1:
            return V + A - A.mean()
        else:
            return V + A - A.mean(dim=1, keepdim=True)
    
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_space)  # acción aleatoria
        else:
            qvals = self.get_qvals(state)  # acción a partir del cálculo del valor de Q para esa acción
            action= torch.max(qvals, dim=-1)[1].item()
        return action
    
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array(state)
        
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.forward(state_t)
    
    def feature_size(self):
        return self.Linear1(autograd.Variable( torch.zeros(1, * self.input_shape)).to(device=self.device)).view(1, -1).size(1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

