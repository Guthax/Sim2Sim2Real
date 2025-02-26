from scenario_builder import  Scenario
from config import _CONFIG_BASIC
scenario = Scenario(_CONFIG_BASIC)
scenario.train_on_environment("duckietown", "duckietown_rgb_cnn", 1000000, 2)
