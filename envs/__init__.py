from gym.envs.registration import register

register(
    id='GaussianBandit-v1',
    entry_point='envs.bandits:BatchBandit',
    kwargs={'max_steps': 10, 'num_actions': 2, 'reward_var': 10, 'mean_var': 100},
)
