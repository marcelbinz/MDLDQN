import gym
import math
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.utils import log_sum_exp
import envs.bandits
from models.mdlrdqn import MDLRDQN
import argparse

def update_target(model, target):
    target.load_state_dict(model.state_dict())

def run(max_steps, N, tau, num_episodes, print_every, save_every, num_hidden, save_dir, vis, prior, return_steps, lr=1e-3, batch_size=16, target_update=100, grad_clip=1.0, gamma=0.96):
    # setup visualization
    if vis:
        from tqdm import trange
        import visdom
        T = trange(int(num_episodes))
        viz = visdom.Visdom()

        def create_vis_plot(_xlabel, _ylabel, _title, _legend):
            return viz.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1, 4)).cpu(),
                opts=dict(
                    xlabel=_xlabel,
                    ylabel=_ylabel,
                    title=_title,
                    legend=_legend
                )
        )

        def update_vis_plot(iteration, nll, kld, reg, window, update_type):
            viz.line(
                X=torch.ones((1, 4)).cpu() * iteration,
                Y=torch.Tensor([nll, kld, nll + kld, reg]).unsqueeze(0).cpu(),
                win=window,
                update=update_type
        )

        plot = create_vis_plot('Iteration', 'Loss', save_dir, ['NLL', 'KLD', 'Loss', 'Regret'])
    else:
        T = range(int(num_episodes))

    # setup environment
    env = gym.make('GaussianBandit-v1')
    env.batch_size = batch_size
    env.max_steps = max_steps

    # setup networks
    model = MDLRDQN(env.observation_space.low.shape[0], env.action_space.n, num_hidden, prior, tau)
    target = MDLRDQN(env.observation_space.low.shape[0], env.action_space.n, num_hidden, prior, tau).eval()
    update_target(model, target)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    performance = []
    nlls = []
    klds = []

    # one episode
    for t in T:
        obs = env.reset()

        hx = model.initial_states(batch_size)
        hx_target = target.initial_states(batch_size)

        zeta = model.get_zeta(batch_size)
        zeta_target = target.get_zeta(batch_size)

        done = False

        performance.append(0)
        nll = 0

        _, _, hx_target = target(obs, hx_target, zeta_target)

        q_distributions = []
        q_targets = []
        rewards = []

        while not done:
            q_distribution, hx, action = model.act(obs, hx, zeta)
            q_distributions.append(q_distribution)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)

            q_target_mean, _, hx_target = target(obs, hx_target, zeta_target)
            if tau > 0.0:
                q_target = log_sum_exp(q_target_mean / tau).squeeze()
            else:
                q_target = q_target_mean.max(1)[0]
            q_targets.append(q_target)

            performance[-1] += info['regrets']

        # compute multi-step returns and negative log-likelihoods
        q_targets[-1].fill_(0)

        for i in range(return_steps - 1):
            q_targets.append(torch.zeros_like(q_targets[-1]))
            rewards.append(torch.zeros_like(rewards[-1]))

        gammas = torch.pow(gamma, torch.arange(return_steps))

        rewards = np.stack(rewards, axis=1)
        for i in range(len(q_distributions)):
            expected_q_value = torch.sum(rewards[:, i:(i+return_steps)] * gammas, dim=1) + (gamma ** return_steps) * q_targets[i + return_steps - 1]
            nll = nll - q_distributions[i].log_prob(expected_q_value.detach()).mean()

        # update parameters
        optimizer.zero_grad()
        nll = nll.div(env.max_steps)
        kld = model.kl_divergence().div(N)
        nlls.append(nll.item())
        klds.append(kld.item())
        (nll + kld).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if (not t % print_every):
            if vis and len(nlls):
                mean_nlls = np.mean(nlls[-print_every:])
                mean_klds = np.mean(klds[-print_every:])
                mean_regrets = np.mean(performance[-print_every:])
                T.set_description('NLL: {:5.2f}, KLD: {:5.2f}, Regret: {:5.2f}'.format(mean_nlls, mean_klds, mean_regrets))
                update_vis_plot(t, mean_nlls, mean_klds, mean_regrets, plot, 'append')

        if (not t % save_every):
            torch.save([N, t, model], save_dir)

        if (not t % target_update):
            update_target(model, target)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--num-episodes', type=int, default=2e5, help='number of trajectories for training')
    parser.add_argument('--print-every', type=int, default=100, help='how often to print')
    parser.add_argument('--save-every', type=int, default=100, help='how often to save')
    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--runs', type=int, default=1, help='total number of runs')

    parser.add_argument('--N', type=int, default=4096, help='number of assumed samples')
    parser.add_argument('--T', type=int, default=10, help='number of steps per episode')
    parser.add_argument('--tau', type=float, default=0.0, help='temperature parameter, if 0 then q-learning is used, else soft q-learning')
    parser.add_argument('--return-steps', type=int, default=4, help='steps for return calculation')

    parser.add_argument('--vis', action='store_true', default=False, help='enable visdom visualization')
    parser.add_argument('--prior', default='grouphs', help='which prior to use')
    parser.add_argument('--save-dir', default='trained_models/', help='directory to save models')

    args = parser.parse_args()
    for i in range(args.runs):
        run(args.T, args.N, args.tau, args.num_episodes, args.print_every, args.save_every, args.num_hidden, args.save_dir + str(i) + '.pt', args.vis, args.prior, args.return_steps)
