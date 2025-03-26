import cv2
import numpy as np
import math
from gymnasium import spaces

from simulators.duckietown import logger
from simulators.duckietown.exceptions import NotInLane
from simulators.duckietown.simulator import Simulator


class DuckietownBaseDynamics(Simulator):
    """
    Wrapper to control the simulator using steering angle
    instead of differential drive motor velocities. Uses RGB camera as observation space and no domain randomization.
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, rgb_camera=True, seg_camera=False, render_img=False, **kwargs):
        self.obs_rgb_name = "camera_rgb"
        self.camera_rgb_enabled = rgb_camera
        self.camera_seg_enabled = seg_camera

        self.laps_completed = 0
        self.laps_done = 0

        self.distance_until_lap_complete = 0.1
        self.min_steps_for_lap = 600
        self.current_steps = 0

        Simulator.__init__(self, **kwargs)

        self.action_space = spaces.Box(low=np.float32(-1), high=np.float32(1))


        if self.camera_rgb_enabled and self.camera_seg_enabled:
            self.observation_space = spaces.Dict({
                "camera_rgb": spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8),
                "camera_seg": spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8),
            })
        else:

            self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3),dtype=np.uint8)

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
        self.render_img = render_img

        self.total_reward = 0
        self.mean_reward = 0
        self.routes_completed = 0
        self.avg_center_dev = 0
        self.avg_speed = 0

        self.first_pos = None




    def convert_steering(self,carla_steering, k=2.5, p=3):
        """Convert CARLA steering values to Duckietown with nonlinear scaling.

        Args:
            carla_steering (float): Steering value from CARLA in range [-1, 1].
            k (float): Scaling factor for amplification.
            p (float): Power factor for nonlinear scaling.

        Returns:
            float: Adjusted steering value for Duckietown in range [-1, 1].
        """
        duckie_steering = np.sign(carla_steering) * k * (abs(carla_steering) ** p)
        return np.clip(duckie_steering, -1, 1)  # Ensure within [-1, 1]

    def step(self, action):
        # Ensure the steering angle is within the valid range
        steering_angle = max(-1, min(1, action))
        #steering_angle = self.convert_steering(action)

        # Map the steering angle to wheel velocities
        left_wheel_velocity = 0.25 * (1 + steering_angle)
        right_wheel_velocity = 0.25 * (1 - steering_angle)

        vels = np.array([left_wheel_velocity, right_wheel_velocity])
        obs_rgb, reward, done, trunc, info = Simulator.step(self, vels)

        steer_value = action
        steer_change_penalty = -abs(steer_value - self.previous_steer) if self.previous_steer else 0
        self.previous_steer = steer_value  # Update previous steering value
        reward += steer_change_penalty

        if done:
            self.laps_done += 1

        if self.camera_rgb_enabled and self.camera_seg_enabled:
            obs_seg = self.render_obs(True)
            obs = {
                "camera_rgb": obs_rgb,
                "camera_seg": obs_seg
            }
        elif self.camera_rgb_enabled:
            obs = obs_rgb
        elif self.camera_seg_enabled:
            obs_seg = self.render_obs(segment=True)
            obs = obs_seg

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
            self.render_images()

        self.current_steps += 1

        distance_to_spawn = abs(np.linalg.norm(self.cur_pos - self.first_pos))
        #print(distance_to_spawn, self.current_steps)
        if distance_to_spawn < self.distance_until_lap_complete and self.current_steps >= self.min_steps_for_lap:
            done = True
            self.laps_completed += 1
            self.laps_done += 1
            # Add a small delay for frame rate control
        return obs, reward, done, trunc, info


    def compute_reward(self, pos, angle, speed):
        """
        try:

            lane_pos = self.get_lane_pos2(pos, angle)

            # Reward for being close to the center of the right lane
            dist_penalty = -abs(lane_pos.dist)  # Negative for larger distances

            # Reward for aligning with the lane direction
            direction_reward = lane_pos.dot_dir  # Higher when aligned with the lane

            # Penalize large misalignment (angle deviation from tangent)
            #angle_penalty = -abs(lane_pos.angle_deg) / 45.0  # Normalize to [-1, 0]
            print(dist_penalty, direction_reward)
            # Compute total reward
            reward = 1.0 + dist_penalty + direction_reward # + angle_penalty

        except NotInLane:
            # Heavy penalty for going out of lane
            reward = -10.0
        """
        col_penalty = self.proximity_penalty2(pos, angle)

        # Get the position relative to the right lane tangent
        try:
            lp = self.get_lane_pos2(pos, angle)


            reward = +1.0  * lp.dot_dir + -10 * np.abs(lp.dist)
            return reward
        except NotInLane:
            return -40

    def reset(self, seed = None, options = None, segment: bool = False):
        obs_rgb, _ = super().reset(seed, options, segment)
        self.first_pos = self.cur_pos
        self.current_steps = 0

        print(f"Laps completed: {self.laps_completed}. Laps done: {self.laps_done}")


        obs_rgb = cv2.cvtColor(obs_rgb, cv2.COLOR_BGR2RGB)
        if self.camera_rgb_enabled and self.camera_seg_enabled:
            obs_seg = self.render_obs(True)
            obs_seg = cv2.cvtColor(obs_seg, cv2.COLOR_BGR2RGB)
            obs = {
                "camera_rgb": obs_rgb,
                "camera_seg": obs_seg
            }
        elif self.camera_rgb_enabled:
            obs = obs_rgb
        elif self.camera_seg_enabled:
            obs_seg = self.render_obs(True)
            return obs_seg, {}

        self.previous_steer = None
        return obs_rgb, {}

    def render_obs(self, segment: bool = False):
        image = Simulator.render_obs(self, segment)
        #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Identify black pixels (where all RGB channels are 0)
        #black_pixels = (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)

        # Generate a single random grey value
        random_grey = np.random.randint(50, 200, dtype=np.uint8)

        # Apply the random grey color to black pixels
        #image[black_pixels] = [255, 255, 255]
        #img = image
        #cv2.imshow("Seg", img)
        #cv2.waitKey(1)
        return image

    def render_images(self):

        if self.camera_rgb_enabled:
            img = self._render_img(
                160,
                120,
                self.multi_fbo_human,
                self.final_fbo_human,
                self.img_array_human,
                top_down=False,
                segment=False,
                custom_segmentation_folder="/home/jurriaan/Documents/Programming/Sim2Sim2Real/simulators/duckietown/segmentation"
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("rgb", img)
            cv2.waitKey(1)
        if self.camera_seg_enabled:
            img = self._render_img(
                160,
                120,
                self.multi_fbo_human,
                self.final_fbo_human,
                self.img_array_human,
                top_down=False,
                segment=True,
                custom_segmentation_folder="/home/jurriaan/Documents/Programming/Sim2Sim2Real/simulators/duckietown/segmentation"
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("seg", img)
            cv2.waitKey(1)


def get_angle(x, y, center_x, center_y):
    return math.atan2(y - center_y, x - center_x)


def has_crossed_spawn(car_x, car_y, spawn_x, spawn_y, center_x, center_y, prev_angle):
    car_angle = get_angle(car_x, car_y, center_x, center_y)
    spawn_angle = get_angle(spawn_x, spawn_y, center_x, center_y)

    # Normalize angles to the range (-pi, pi)
    diff = car_angle - spawn_angle

    # Detect crossing by checking if the angle difference changes sign
    crossed = (prev_angle < spawn_angle and car_angle > spawn_angle) or \
              (prev_angle > spawn_angle and car_angle < spawn_angle)

    return crossed, car_angle