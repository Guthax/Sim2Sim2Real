from stable_baselines3.common.callbacks import CheckpointCallback
import config
from evaluator import Evaluator

config.set_config("TEST")
from envs.carla.carla_route_env import CarlaRouteEnv
from simulators.carla.carla_env.rewards import reward_functions
from simulators.carla.carla_env.state_commons import create_encode_state_fn, load_vae
from trainer import Trainer
from stable_baselines3 import PPO


def evaluate():
    # F:\Programs\CARLA\CARLA913\CarlaUE4.exe -quality-level=Low -benchmark -fps=15 -prefernvidia -dx11 -carla-world-port=2000
    from config import CONFIG

    configuration = CONFIG
    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(None, configuration["state"])
    env = CarlaRouteEnv(obs_res=configuration["obs_res"], host="localhost", port=2000,
                        reward_fn=reward_functions[configuration["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                        fps=15, action_smoothing=configuration["action_smoothing"],
                        action_space_type='continuous', activate_spectator=False, activate_render=True)
    algorithm = PPO
    algorithm = algorithm.load("F:\\Github\\Sim2Sim2Real\\results\\carla_only_rgb_model_trained_1000.zip", env=env, device='cuda')


    evaluator = Evaluator(env, algorithm)

    evaluator.evaluate()


evaluate()