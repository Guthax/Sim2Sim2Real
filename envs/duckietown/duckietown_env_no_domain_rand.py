import cv2
import numpy as np
from gym import spaces

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

        self.domain_rand = False
        self.camera_rand = False
        self.dynamics_rand = False

        self.render_img = render_img

    def step(self, action):
        vel, angle = 0.1, action
        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done, info = Simulator.step(self, vels)
        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        info["DuckietownEnv"] = mine

        if self.render_img:
            self.render(mode='human')

        return obs, reward, done, info
