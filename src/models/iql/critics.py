import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config["env"]["state_dim"] + \
            config["env"]["action_dim"]
        self.output_dim = 1
        self.intermediate_dim = config["intermediate_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.Mish(),

            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.Mish(),

            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.Mish(),

            nn.Linear(self.intermediate_dim, self.output_dim)
        )

    def forward(self, state, action):
        return self.mlp(torch.cat([state, action], dim=-1))


class VNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config["env"]["state_dim"]
        self.output_dim = 1
        self.intermediate_dim = config["intermediate_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.Mish(),

            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.Mish(),

            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.Mish(),

            nn.Linear(self.intermediate_dim, self.output_dim)
        )

    def forward(self, state) -> torch.Tensor:
        return self.mlp(state)