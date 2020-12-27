
import torch
from torch import nn

class BaselineSpiralClassifier(nn.Module):
    """
    Baseline architecture for the spiral classifier. Consists of 6 input units, 8 hidden units, and 1 output unit.
    """

    def __init__(self):
        super(BaselineSpiralClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 1)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class LinearInputsSpiralClassifier(nn.Module):
    """
    Neural network spiral classifier with only linear input features.
    """

    def __init__(self):
        super(LinearInputsSpiralClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class GenericSpiralClassifier(nn.Module):
    """
    Generic neural network Neural network, capabale of taking a network structure and optionally a list of activation functions.
    By default the nonlinearities in the network are tan(x) for each layer.

    Choose from these three nonlinearities for each layer:
        - A: Tan
        - B: ReLU
        - C: Sigmoid

    The length of the activations must be the same as the number of hidden layers in the network structure.

    Example:
        - To create a neural network with:
            - 6 input features
            - 3 hidden layers, each with 8 units and tan, relu, and sigmoid activations respectively
            - 1 output feature.
        - Provide these arguments:
            - network_structure = [6, 8, 8, 8, 1]
            - nonlinearity_keys = ["A", "B", "C"]
    """
    
    def __init__(self, network_structure, nonlinearity_keys=None):
        super(GenericSpiralClassifier, self).__init__()
        
        nonlinearity_dict = {"A": torch.tanh, "B": torch.relu, "C": torch.sigmoid}

        assert (len(network_structure) - 2 == len(nonlinearity_keys))
        nonlinearities = [nonlinearity_dict[nonlinearity_keys[i]] for i in range(len(nonlinearity_keys))]
            
        self.nonlinearities = nonlinearities

        self.layers = nn.ModuleList()
        for i in range(len(network_structure) - 1):
            self.layers.append(nn.Linear(network_structure[i], network_structure[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.nonlinearities[i](layer(x))
        x = self.layers[-1](x)
        return x
