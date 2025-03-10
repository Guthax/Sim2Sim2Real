import numpy as np
import torch
from stable_baselines3.common.noise import NormalActionNoise

from utils import lr_schedule
import torch as th

algorithm_params = {
    "PPO": dict(
        #learning_rate=lr_schedule(1e-4, 1e-6, 2),
        learning_rate=lr_schedule(3e-4, 1e-5, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=2048,
        #use_sde=True,
        #sde_sample_freq=4,
        policy_kwargs=dict(
            #net_arch=dict(pi=[500, 300], vf=[500, 300]),  # Larger network for better feature extraction
            net_arch=dict(pi=[1024, 512, 256], vf=[1024, 512, 256]),
            activation_fn=torch.nn.ReLU,  # ReLU activation for stable gradients
            log_std_init=-2,  # Lower initial std to encourage smaller actions
        )
        #policy_kwargs=dict(activation_fn=th.nn.ReLU,
        #                   net_arch=[dict(pi=[500, 300], vf=[500, 300])])
    ),
    "SAC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    "DDPG": dict(
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    "SAC_BEST": dict(
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[500, 300]),
    ),
}