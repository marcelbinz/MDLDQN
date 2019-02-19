import torch
import torch.nn as nn
import torch.nn.functional as F
from models.linear import LinearGroupHS

class GroupHSGRUCell(torch.nn.modules.rnn.RNNCellBase):

    def __init__(self, input_size, hidden_size):
        super(GroupHSGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = LinearGroupHS(input_size, 3 * hidden_size)
        self.weight_hh = LinearGroupHS(hidden_size, 3 * hidden_size)

    def step(self, input, hidden, zeta):

        gi = self.weight_ih(input, zeta[0])
        gh = self.weight_hh(hidden, zeta[1])

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy

    def forward(self, input, hx, zeta):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx)

        return self.step(input, hx, zeta)

    def get_zeta(self, batch_size):
        # first for W_ih, second for W_hh
        return (self.weight_ih.get_zeta(batch_size), self.weight_hh.get_zeta(batch_size))

    def kl_divergence(self):
        return self.weight_ih.kl_divergence() + self.weight_hh.kl_divergence()
