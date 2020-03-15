import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

class DNN(nn.Module):
    """
	Deep Neural networks (DNN)
    """
    def __init__(self, input_size, hidden_size, out_size, dropout_drop=0.5):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size, out_size)
        self.tanh = nn.Tanh()
        #init weights
        self.init_weights()
    def init_weights(self):
        """
        Initialize weights for fully connected layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
    def forward(self, x):
        """
        Forward pass of DNN.
        Args:
            x: input batch (signal)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        x = self.tanh(x)
        return x
         
