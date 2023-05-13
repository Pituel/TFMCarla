import os
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR

import numpy as np
import torch.autograd as autograd


class DuelingDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(DuelingDQnetwork, self).__init__()
        self.input_shape = (95 + 5,)
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR, model)
        self.action_space = np.arange(self.n_actions)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # self.Linear1 = nn.Sequential(
        #     nn.Linear(95 + 5, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU()
        # )

        self.Linear1 = nn.Sequential(
            nn.Linear(95 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        if torch.cuda.is_available():
            self.Linear1.cuda()
        
        self.fc_layer_inputs = self.feature_size()
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

        if self.device == 'cuda':
            self.V.cuda()
            self.A.cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=DQN_LEARNING_RATE)


    def forward(self, x):
        print(x)
        fc = self.Linear1(x)
        print(fc)
        V = self.V(fc)
        A = self.A(fc)
        return V + A - A.mean()
    
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

