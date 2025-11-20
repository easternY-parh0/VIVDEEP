import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden=256, nlayers=4, out_dim=4):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(nlayers):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.Tanh())
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

