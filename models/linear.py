import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import  Normal, Laplace, LogNormal
from torch.nn import init, Parameter

class LinearGroupHS(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearGroupHS, self).__init__()
        self.in_features = in_features + 1
        self.out_features = out_features

        self.tau_0 = 1e-5

        self.sa_mu = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.sa_logvar = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.sb_mu = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.sb_logvar = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.alpha_mu = nn.Parameter(torch.Tensor(self.in_features), requires_grad=True)
        self.alpha_logvar = nn.Parameter(torch.Tensor(self.in_features), requires_grad=True)

        self.beta_mu = nn.Parameter(torch.Tensor(self.in_features), requires_grad=True)
        self.beta_logvar = nn.Parameter(torch.Tensor(self.in_features), requires_grad=True)

        self.weight_mu = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)
        self.weight_logvar = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)

        self.reset_parameters()

        # numerical stability param
        self.epsilon = 1e-8

    def reset_parameters(self):
        self.sa_mu.data.normal_(-3, 1e-2)
        self.sb_mu.data.normal_(1, 1e-2)
        self.alpha_mu.data.normal_(1, 1e-2)
        self.beta_mu.data.normal_(1, 1e-2)

        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.normal_(0, stdv)

        # init logvars
        self.sa_logvar.data.normal_(-6, 1e-2)
        self.sb_logvar.data.normal_(-6, 1e-2)
        self.alpha_logvar.data.normal_(-6, 1e-2)
        self.beta_logvar.data.normal_(-6, 1e-2)
        self.weight_logvar.data.normal_(-6, 1e-2)


    def forward(self, x, zeta):
        b = torch.ones(x.shape[0], 1)
        return F.linear(torch.cat((x, b), dim=1), zeta)

    def kl_divergence(self):
        KLD = torch.zeros(1)

        # KL(q(z)||p(z))
        KLD += -math.log(self.tau_0) + (torch.exp(self.sa_mu + 0.5 * self.sa_logvar.exp()) / self.tau_0) - 0.5 * (self.sa_mu + self.sa_logvar + 1 + math.log(2.0))
        KLD += torch.exp(0.5 * self.sb_logvar.exp() - self.sb_mu) - 0.5 * (-self.sb_mu + self.sb_logvar + 1 + math.log(2.0))

        KLD_element = -1.0 * (-torch.exp(self.alpha_mu + 0.5 * self.alpha_logvar.exp()) + 0.5 * (self.alpha_mu + self.alpha_logvar + 1 + math.log(2.0)))
        KLD += torch.sum(KLD_element)

        KLD_element = -1.0 * (-torch.exp(0.5 * self.beta_logvar.exp() - self.beta_mu) + 0.5 * (-self.beta_mu + self.beta_logvar + 1 + math.log(2.0)))
        KLD += torch.sum(KLD_element)

        # KL(q(w|z)||p(w|z))
        # we use the kl divergence given by [3] Eq.(8)
        KLD_element = -0.5 * self.weight_logvar + 0.5 * (self.weight_logvar.exp() + self.weight_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)


        return KLD

    def get_zeta(self, batch_size):
        sa = LogNormal(self.sa_mu, torch.sqrt(self.sa_logvar.exp()))
        sb = LogNormal(self.sb_mu, torch.sqrt(self.sb_logvar.exp()))
        alpha = LogNormal(self.alpha_mu, torch.sqrt(self.alpha_logvar.exp()))
        beta = LogNormal(self.beta_mu, torch.sqrt(self.beta_logvar.exp()))
        weights = Normal(self.weight_mu, torch.sqrt(self.weight_logvar.exp()))

        if self.training:
            return torch.sqrt(sa.rsample() * sb.rsample() * alpha.rsample() * beta.rsample()) * weights.rsample()
        else:
            return torch.sqrt(sa.mean * sb.mean * alpha.mean * beta.mean) * weights.mean

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
