import torch
import torch.nn as nn
from torch.distributions import  Normal, Categorical
from models.linear import LinearGroupHS
from models.recurrent import GroupHSGRUCell

class MDLRDQN(nn.Module):
    def __init__(self, num_states, num_actions, num_hidden, prior, tau):
        super(MDLRDQN, self).__init__()
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        self.tau = tau

        self.initial = nn.Parameter(0.01 * torch.randn(1, self.num_hidden), requires_grad=True)

        if (prior == 'grouphs'):
            self.gru = GroupHSGRUCell(num_states, num_hidden)
            self.mu = LinearGroupHS(num_hidden, num_actions)


    def forward(self, input, hx, zeta):
        hx = self.gru(input, hx, zeta[0])

        return self.mu(hx, zeta[1]), None, hx

    def act(self, input, hx, zeta):
        q_values_mean, q_values_scale, hx = self(input, hx, zeta)

        if self.tau > 0.0:
            normalized_policy = Categorical(logits=nn.functional.log_softmax(q_values_mean / self.tau, dim=1))
            action = normalized_policy.sample()
        else:
            action = q_values_mean.max(1)[1]

        q_value_mean = q_values_mean.gather(1, action.unsqueeze(1)).squeeze(1)

        return Normal(q_value_mean, 10), hx, action

    def initial_states(self, batch_size):
        return self.initial.expand(batch_size, -1)

    def get_zeta(self, batch_size):
        return (self.gru.get_zeta(batch_size), self.mu.get_zeta(batch_size))

    def kl_divergence(self):
        return self.gru.kl_divergence() + self.mu.kl_divergence()
