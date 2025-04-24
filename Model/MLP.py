import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation = 'relu',dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size



        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_size))  
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        return self.model(x)