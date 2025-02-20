import cv2
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

from envs.duckietown.duckietown_env import DuckietownEnv
from simulators.duckietown.simulator import Simulator
from simulators.duckietown import logger


class DuckietownDirectVelocities(DuckietownEnv):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, render_img=False, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)
        logger.info("using DuckietownEnvNoDomainrand")

        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "rgb_camera": self.observation_space
        })

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit
        self.distortion = True
        self.domain_rand = True
        self.camera_rand = True
        self.dynamics_rand = True
        self.render_img = render_img

        self.total_reward = 0
        self.mean_reward = 0
        self.routes_completed = 0
        self.avg_center_dev = 0
        self.avg_speed = 0


    def step(self, action):
        # Ensure the steering angle is within the valid range
        steering_angle = max(-1, min(1, action))

        # Map the steering angle to wheel velocities
        left_wheel_velocity = 0.25 * (2 + steering_angle)
        right_wheel_velocity = 0.25 * (2 - steering_angle)

        vels = np.array([left_wheel_velocity, right_wheel_velocity])

        obs, reward, done, info = Simulator.step(self, vels)
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.step_count


        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = left_wheel_velocity
        mine["omega_l"] = right_wheel_velocity

        info["DuckietownEnv"] = mine
        info["total_reward"] = self.total_reward
        info["routes_completed"] = self.routes_completed
        info["total_distance"] = 0
        info["avg_center_dev"] = self.avg_center_dev
        info["avg_speed"] = self.avg_speed
        info["mean_reward"] = self.mean_reward
        info["completed_steps"] = self.step_count

        if self.render_img:
            img = self.render(mode='top_down')
            #img = cv2.flip(img, 0)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #canny = cv2.Canny(img, 100, 200)


            cv2.imshow('output', img)
            cv2.waitKey(1)

            # Add a small delay for frame rate control
        return obs, reward, done, info
