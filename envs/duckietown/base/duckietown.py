import cv2
import numpy as np
from gym import spaces

from envs.duckietown.duckietown_env import DuckietownEnv
from simulators.duckietown import logger
from simulators.duckietown.exceptions import NotInLane
from simulators.duckietown.simulator import Simulator


class DuckietownBaseDynamics(Simulator):
    """
    Wrapper to control the simulator using steering angle
    instead of differential drive motor velocities. Uses RGB camera as observation space and no domain randomization.
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, render_img=False, **kwargs):
        self.obs_rgb_name = "camera_rgb"
        Simulator.__init__(self, **kwargs)

        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

        self.observation_space = spaces.Dict({
            self.obs_rgb_name : spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8)
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
        self.distortion = False
        self.domain_rand = False
        self.camera_rand = False
        self.dynamics_rand = False
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
            img = self._render_img(
                800,
                600,
                self.multi_fbo_human,
                self.final_fbo_human,
                self.img_array_human,
                top_down=True,
                segment=False,
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("top_down_view", img)
            cv2.waitKey(1)

            # Add a small delay for frame rate control
        obs = {
            self.obs_rgb_name: obs
        }
        return obs, reward, done, info


    def compute_reward(self, pos, angle, speed):
        try:
            lane_pos = self.get_lane_pos2(pos, angle)

            # Reward for being close to the center of the right lane
            dist_penalty = -abs(lane_pos.dist)  # Negative for larger distances

            # Reward for aligning with the lane direction
            direction_reward = lane_pos.dot_dir  # Higher when aligned with the lane

            # Penalize large misalignment (angle deviation from tangent)
            angle_penalty = -abs(lane_pos.angle_deg) / 45.0  # Normalize to [-1, 0]

            # Compute total reward
            reward = 1.0 + dist_penalty + direction_reward + angle_penalty

        except NotInLane:
            # Heavy penalty for going out of lane
            reward = -10.0
        return reward

    def reset(self, segment: bool = False):
        obs = super().reset()
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        return {
            self.obs_rgb_name: obs
        }

    def render_obs(self, segment: bool = False):
        img = Simulator.render_obs(self, segment)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img