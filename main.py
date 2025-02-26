from util.scenario import  Scenario
from config.scenario_configs import _CONFIG_PPO_CNN_DUCKIETOWN_RGB_STEER, _CONFIG_PPO_MULTI_DUCKIETOWN_RGB_CANNY_STEER

import argparse

parser = argparse.ArgumentParser("runner")
parser.add_argument("config_name", help="The name of the config to use.", type=str)
parser.add_argument("env_name", help="Environment to train/evaluate on", t)
args = parser.parse_args()
print(args.counter + 1)


scenario = Scenario(_CONFIG_PPO_MULTI_DUCKIETOWN_RGB_CANNY_STEER)
scenario.train_on_environment("duckietown",
                              "duckietown_multi_rgb_canny",
                              "results",
                              "tensorboard",
                              1000000, 5)


#scenario.evaluate_on_environment("duckietown", "/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckietown_rgb_cnn_model_trained_1000.zip")