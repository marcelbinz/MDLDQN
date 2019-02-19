import torch
import torch.nn as nn
import numpy as np
from torch.distributions import  Normal, Bernoulli
from torch.autograd import grad
import statsmodels.api as sm

class LaplaceProbitRegression(nn.Module):
    def __init__(self, in_features):
        super(LaplaceProbitRegression, self).__init__()
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, 1)
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x):
        return Normal(torch.zeros(1), torch.ones(1)).cdf(self.linear(x)).squeeze()

    def fit(self, x, y):
        sm_probit_link = sm.genmod.families.links.probit
        glm_binom = sm.GLM(y, x, family=sm.families.Binomial(link=sm_probit_link))
        params = glm_binom.fit().params
        self.linear.weight[0] = torch.from_numpy(params)

        return params


    def hessian(self, x, y):
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        # compute log P(D|w)
        log_likelihood = Bernoulli(self(x)).log_prob(y).sum()

        # compute log P(w)
        log_prior = Normal(torch.zeros(self.linear.weight.shape), torch.ones(self.linear.weight.shape)).log_prob(self.linear.weight).sum()
        # compute -log P(D, w) = -log P(D|w) - log P(w)
        E = -log_likelihood - log_prior

        # compute hessian
        x_1grad, = grad(E, self.linear.weight, create_graph=True)
        hessian = []
        for i in range(x_1grad.size(1)):
            x_2grad, = grad(x_1grad[0, i], self.linear.weight, create_graph=True)
            hessian.append(x_2grad[0])

        hessian = torch.stack(hessian)

        # return diag of standard deviations
        return np.sqrt(np.diag(torch.inverse(hessian).detach().numpy()))


if __name__ == "__main__":
    N = 100
    D = 3

    pr = LaplaceProbitRegression(D)
    x = torch.zeros(N, D).normal_(0, 1)
    y = torch.zeros(N).bernoulli_(0.5)
    sigma = pr.hessian(x, y)
    print(sigma)
