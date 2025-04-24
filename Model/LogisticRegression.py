import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):

        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
