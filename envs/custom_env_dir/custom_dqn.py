import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime

class DeepQNetwork(nn.Module):
    
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, fc1_dims, fc2_dims, seed, optimizer):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.checkpoint_file_final = os.path.join(self.checkpoint_dir, str(name+'_final'))
        
        # Sets the seed for generating random numbers (here initial weights and biases?)
        self.seed = T.manual_seed(seed)
            
        # linear input connection from input to first hidden layer neurons
        # applies a linear transformation to the incoming data: y = xA^T + b
        self.fc1 = nn.Linear(input_dims,fc1_dims)
        
        # linear input connection from the hidden layer to output
        self.fc3 = nn.Linear(fc1_dims,n_actions)
        
        # Allows you to select optimizer with a string in the main method
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr) 
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        observation = T.Tensor(state).to(self.device)
        x = F.relu(self.fc1(observation))
        actions = self.fc3(x)

        return actions

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def save_checkpoint_final(self):
        #print('... saving final checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file_final)

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
        
    def load_checkpoint_final(self):
        #print('... loading final checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file_final))