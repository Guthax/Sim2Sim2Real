from util.scenario import  Scenario
from config.scenario_configs import _CONFIG_BASIC

scenario = Scenario(_CONFIG_BASIC)
scenario.train_on_environment("duckietown",
                              "duckietown_rgb_cnn",
                              "results",
                              "tensorboard",
                              1000, 5)


#scenario.evaluate_on_environment("duckietown", "/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckietown_rgb_cnn_model_trained_1000.zip")