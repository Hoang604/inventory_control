import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, config) -> nn.Module:
        super().__init__()
        self.input_dim = config['env']['state_dim']
        self.intermediate_dim = config['intermediate_dim']
        self.output_dim = config['actor']['output_dim']
        self.log_std_min = config['actor']['log_std_min']
        self.log_std_max = config['actor']['log_std_max']

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

    def forward(self, state):
        mean, log_std = torch.chunk(self.mlp(state), chunks=2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        action = dist.rsample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean