import numpy as np
import torch
from stable_baselines3.common.noise import NormalActionNoise

from util.custom_extractors.parsing_net.extractor import ParsingNetFeatureExtractor
from util.custom_extractors.resnet.resnet8_extractor import ResnetExtractor
from util.feature_extractors import PaperCNN
from stable_baselines3.common.torch_layers import CombinedExtractor, NatureCNN
from utils import lr_schedule
import torch as th

algorithm_params = {
    "PPO": dict(
    learning_rate=lr_schedule(3e-4, 1e-5, 3),
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=1e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=42,
    policy_kwargs=dict(
            net_arch=[512, 256],
            activation_fn=th.nn.ReLU,
            normalize_images=False,
            log_std_init = -1,
            features_extractor_class=CombinedExtractor,
            features_extractor_kwargs=dict(normalized_image=True, cnn_output_dim=512)
    ),
    ),
    "PPO_FEATURE": dict(
        learning_rate=lr_schedule(3e-4, 1e-5, 2),  # Tunable; schedules can help too
        n_steps=1024,  # Larger buffer for image-based learning
        batch_size=256,  # Larger mini-batch for stable gradients
        n_epochs=10,  # Fewer epochs for large batch/step sizes
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # Policy clip range
        ent_coef=0.02,  # Entropy regularization (beta)
        vf_coef=0.5,  # Value loss weight
        max_grad_norm=0.5,  # Gradient clipping
        policy_kwargs=dict(
            net_arch=[512, 256],  # 2 layers of 256 units (Unity: 32–512)
            activation_fn=torch.nn.ReLU,
            log_std_init=-1,
            features_extractor_class=ResnetExtractor
        ),
    ),
    """
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
            log_std_init=-1,  # Lower initial std to encourage smaller actions
        )
        #policy_kwargs=dict(activation_fn=th.nn.ReLU,
        #                   net_arch=[dict(pi=[500, 300], vf=[500, 300])])
    ),
    """
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
        learning_rate=0.0003,
        buffer_size=1000,  # Can increase to 1M if memory allows
        learning_starts=1000,  # Delay learning to allow buffer to fill
        batch_size=256,  # 128–256 is common
        tau=0.005,  # Target smoothing coefficient
        gamma=0.99,  # Discount factor
        train_freq=8,  # Train every step
        gradient_steps=8,  # One update per step (can increase if you want more updates per env step)
        ent_coef="auto",  # Automatically tuned entropy regularization
        use_sde=True,  # Use State-Dependent Exploration (helps on sim2real)
        sde_sample_freq=4,  # How often to resample SDE noise
    ),
    "TD3": dict(
        policy_kwargs= {
            "features_extractor_class": PaperCNN,
            "features_extractor_kwargs": {"features_dim": 256},
        },
        learning_rate= 0.0001,
        buffer_size= 300000,
        learning_starts= 10000,
        batch_size= 100,
        tau= 0.01,
        gamma= 0.99,
        train_freq= 1000,
        gradient_steps= 1000,
        policy_delay= 2,
        target_policy_noise= 0.2,
        target_noise_clip= 0.5,
        action_noise= NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1)),  # Encourages exploration
    )

}
