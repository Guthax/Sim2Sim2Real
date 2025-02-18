import cv2
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

from envs.duckietown.duckietown_env import DuckietownEnv
from simulators.duckietown.simulator import Simulator
from simulators.duckietown import logger


class DuckietownEnvNoDomainRand(DuckietownEnv):
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




    def step(self, action):
        vel, angle = 0.1, action
        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain - self.trim) / k_r
        k_l_inv = (self.gain + self.trim) / k_l

        omega_r = (vel - 0.5 * angle * baseline) / self.radius
        omega_l = (vel + 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done, info = Simulator.step(self, vels)
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.step_count
        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l

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
