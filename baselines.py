import math
import random
import torch

class ValueDirected:
    def act(self, priors, obs=None, hx=None, zeta=None):
        x = priors[1, 0] - priors[0, 0]
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

class ThompsonSampling:
    def act(self, priors, obs=None, hx=None, zeta=None):
        x = (priors[1, 0] - priors[0, 0]) / (math.sqrt(priors[1, 1] + priors[0, 1]))
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

class UCB:
    def __init__(self):
        self.gamma = 1.0
        self.lamb = 1.0

    def act(self, priors, obs=None, hx=None, zeta=None):
        x = ((priors[1, 0] - priors[0, 0]) + self.gamma * (math.sqrt(priors[1, 1]) - math.sqrt(priors[0, 1]))) / self.lamb
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

class Greedy:
    def act(self, priors, obs=None, hx=None, zeta=None):
        x = priors[1, 0] - priors[0, 0]

        if x == 0:
            return random.choice([0, 1])
        else:
            return 1 if (x > 0) else 0

class Optimal:
    def __init__(self):
        _, _, self.model = torch.load('trained_models/infty/0.pt')

    def act(self, priors, obs, hx, zeta):
        action = 0
        for m in range(len(hx)):
            q_mu, _, hx[m] = model(obs, hx[m], zeta[m])
            action += torch.argmax(q_mu[0]).float().item()

        action_model = action_model / len(hx)

        return 1 if (action_model > 0.5) else 0
