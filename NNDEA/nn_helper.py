
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn


# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.hidden_0 = nn.Linear(1, 64)
        self.hidden_1 = nn.Linear(64, 128)
        self.hidden_2 = nn.Linear(128, 128)
        self.hidden_3 = nn.Linear(128, 64)
        self.hidden_4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 5)
        self.activation = nn.ReLU6()

    def forward(self, t):
        t = t.view(-1, 1).float()
        x = self.activation(self.hidden_0(t))
        x = self.activation(self.hidden_1(x))
        x = self.activation(self.hidden_2(x))
        x = self.activation(self.hidden_3(x))
        x = self.activation(self.hidden_4(x))
        y = self.output(x)
        return (1-torch.exp(-t))*y
    

def plus(x):
  return x if x > 0 else 0



if __name__ == "__main__":
    # Generate training data
    t_values = torch.linspace(10, 12, 20).reshape(-1, 1)
    y_initial = torch.tensor([0.0] * 5)  # y(0) = 1







   

