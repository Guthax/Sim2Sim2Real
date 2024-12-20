from stable_baselines3.common.callbacks import CheckpointCallback
import config
from envs.carla.carla_steering_only_env import CarlaSteeringEnv

config.set_config("TEST")
from envs.carla.carla_route_env import CarlaRouteEnv
from simulators.carla.carla_env.rewards import reward_functions
from simulators.carla.carla_env.state_commons import create_encode_state_fn, load_vae
from trainer import Trainer
from stable_baselines3 import PPO


def train():

    from config import CONFIG

    configuration = CONFIG
    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(None, configuration["state"])
    """
    env = CarlaRouteEnv(obs_res=configuration["obs_res"], host="localhost", port=2000,
                        reward_fn=reward_functions[configuration["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                        fps=15, action_smoothing=configuration["action_smoothing"],
                        action_space_type='continuous', activate_spectator=False, activate_render=True)
    """
    env = CarlaSteeringEnv(obs_res=configuration["obs_res"], host="localhost", port=2000,
                        reward_fn=reward_functions[configuration["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                        fps=15, action_smoothing=configuration["action_smoothing"],
                        action_space_type='continuous', activate_spectator=False, activate_render=True)
    algorithm = PPO('MultiInputPolicy', env, verbose=2, device='cuda', **configuration["algorithm_params"])

    trainer = Trainer(env, algorithm)

    num_timesteps = 100000
    num_checkpoints = 5

    save_path = "../../results"
    trainer.train("carla_only_rgb_steering", num_timesteps, [], 5, save_path)

train()