import argparse

from config.scenario_configs import configs, get_config_by_name
from util.scenario import Scenario

parser = argparse.ArgumentParser("evaluator")
parser.add_argument("-config_name", help="The name of the config to use.", type=str)
parser.add_argument("-env_name", help="Environment to evaluate on", type=str)
parser.add_argument("-model_path", help="Path to model weights", type=str)

args = parser.parse_args()


scenario = Scenario(get_config_by_name(args.config_name))

scenario.evaluate_on_environment(args.env_name, args.model_path, True)

# python evaluate.py -config_name duckietown_rgb_test -env_name duckietown -model_path /home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckietown_rgb_cnn_model_trained_400000_steps.zip
