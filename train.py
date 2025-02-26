import argparse

from config.scenario_configs import configs, get_config_by_name
from util.scenario import Scenario

parser = argparse.ArgumentParser("evaluator")
parser.add_argument("-config_name", help="The name of the config to use.", type=str)
parser.add_argument("-env_name", help="Environment to train on", type=str)
parser.add_argument("-model_name", help="Model output name", type=str)
parser.add_argument("-output", help="Output directory", type=str, default="results")
parser.add_argument("-log_dir", help="Tensorboard logdirectory", type=str, default="tensorboard")
parser.add_argument("-steps", help="timesteps to train", type=int, default=1000000)
parser.add_argument("-checkpoints", help="amount of checkpoints", type=int, default=5)

args = parser.parse_args()


scenario = Scenario(get_config_by_name(args.config_name))

scenario.train_on_environment(args.env_name,
                              args.model_name,
                              args.output,
                              args.log_dir,
                              args.steps,
                              args.checkpoints)

# python train.py -config_name scenario_1 -env_name duckietown -model_name duckietown_rgb_cnn

