from scenario_builder import make_scenario_from_config
from config import _CONFIG_BASIC
scenario = make_scenario_from_config(_CONFIG_BASIC)
scenario.train_on_environment("duckietown")

