from typing import Any

import cv2
import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperObsType
from stable_baselines3.common.utils import obs_as_tensor
from torchvision.transforms import transforms


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, dst_width=160, dst_height=120):
        super().__init__(env)
        self.resize_h = dst_height
        self.resize_w = dst_width

        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = self._resize_dict_space(env.observation_space)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.resize_h, self.resize_w, 3),
                                                dtype=np.uint8)

    def _resize_dict_space(self, obs_space):
        new_spaces = {}
        for key, space in obs_space.spaces.items():
            if isinstance(space, spaces.Box) and len(space.shape) == 3:
                new_spaces[key] = spaces.Box(low=0, high=255, shape=(self.resize_h, self.resize_w, 3), dtype=np.uint8)
            else:
                new_spaces[key] = space  # Keep non-image spaces unchanged
        return spaces.Dict(new_spaces)

    def observation(self, observation):
        if isinstance(observation, dict):
            return {key: (cv2.resize(val, (self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)
                          if isinstance(val, np.ndarray) and len(val.shape) == 3 else val)
                    for key, val in observation.items()}
        else:
            return cv2.resize(observation, (self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)

class SegmentationFilterWrapper(gym.ObservationWrapper):
    colors_to_keep_seg = [
        (128, 64, 128),
        (157, 234, 50)
    ]
    reset_every = 10000

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self.gray_value = np.random.randint(0, 256, dtype=np.uint8)
        self.counter = 0
        #window = cv2.namedWindow("filtered")

    def reset(self, **kwargs):
        """Reset environment and randomize gray background."""
        
        if self.counter >= self.reset_every:
            self.gray_value = np.random.randint(0, 256, dtype=np.uint8)
            self.counter = 0  # Generate gray value on reset
        return super().reset(**kwargs)

    def observation(self, observation):
        array = observation
        if isinstance(self.env.observation_space, spaces.Dict):
            array = observation["camera_seg"]

        mask = np.zeros(array.shape[:2], dtype=np.uint8)
        for color in self.colors_to_keep_seg:
            mask |= np.all(array == color, axis=-1)  # Mark pixels that match any of the colors

        # Apply the mask: Keep only selected colors, set others to black
        filtered_image = np.full_like(array, self.gray_value)  # Create a black image
        filtered_image[mask == 1] = array[mask == 1]  # Copy only the kept colors
        cv2.imshow("filtered", filtered_image)
        cv2.waitKey(1)
        self.counter += 1

        result = observation
        if isinstance(self.env.observation_space, spaces.Dict):
            result["camera_seg"] = filtered_image
        else:
            result = filtered_image

        return result


class CannyWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(60, 160, 1), dtype=np.uint8)

    def observation(self, observation):
        processed_image = self.detect_lanes(observation)
        processed_image = np.expand_dims(processed_image, axis=2)  # Expand dimensions for compatibility
        return processed_image

    def detect_lanes(self, image):
        gray = self.grayscale(image)
        blur_gray = self.gaussian_blur(gray, 5)
        edges = self.canny(blur_gray, 50, 150)
        imshape = image.shape
        vertices = np.array([[(0, imshape[0]), (2.4*imshape[1]/5, 1.22*imshape[0]/2),
                              (2.6*imshape[1]/5, 1.22*imshape[0]/2), (imshape[1], imshape[0])]], dtype=np.int32)
        masked_edges = self.region_of_interest(edges, vertices)
        return  edges
        #rho, theta, threshold, min_line_len, max_line_gap = 2, np.pi/180, 15, 40, 200
        #line_image = self.hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
        #result = self.weighted_img(line_image, image)
        #return result

    @staticmethod
    def grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def canny(img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    @staticmethod
    def gaussian_blur(img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    @staticmethod
    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        ignore_mask_color = 255 if len(img.shape) == 2 else (255,) * img.shape[2]
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)

    @staticmethod
    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        CannyWrapper.draw_lines(line_img, lines)
        return line_img

    @staticmethod
    def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
        return cv2.addWeighted(initial_img, α, img, β, γ)

    @staticmethod
    def extrapolate(x, y):
        z = np.polyfit(x, y, 1)
        f = np.poly1d(z)
        x_new = np.linspace(min(x), max(x), 10).astype(int)
        y_new = f(x_new).astype(int)
        return x_new[0], y_new[0], x_new[-1], y_new[-1]

    @staticmethod
    def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
        if lines is None:
            return
        xp, yp, xn, yn = [], [], [], []
        for line in lines:
            for x1, y1, x2, y2 in line:
                m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                if m > 0.5:
                    xp += [x1, x2]
                    yp += [y1, y2]
                elif m < -0.5:
                    xn += [x1, x2]
                    yn += [y1, y2]
        if xp:
            pxp, pyp, cxp, cyp = CannyWrapper.extrapolate(xp, yp)
            if abs((cyp - pyp) / (cxp - pxp)) > 0.5:
                cv2.line(img, (pxp, pyp), (cxp, cyp), color, thickness)
        if xn:
            pxn, pyn, cxn, cyn = CannyWrapper.extrapolate(xn, yn)
            if abs((cyn - pyn) / (cxn - pxn)) > 0.5:
                cv2.line(img, (pxn, pyn), (cxn, cyn), color, thickness)


class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, crop_height_start= 60, crop_height_end=120, crop_width_start=0, crop_width_end=160, channels=3):
        gym.ObservationWrapper.__init__(self, env)

        self.crop_h_start =crop_height_start
        self.crop_h_end = crop_height_end
        self.crop_w_start = crop_width_start
        self.crop_w_end = crop_width_end
        self.num_channels = channels
        self.observation_space = spaces.Box(low=0, high=255, shape=(crop_height_end - crop_height_start, crop_width_end - crop_width_start, channels), dtype=np.uint8)

    def observation(self, observation):
        cropped = observation[self.crop_h_start:self.crop_h_end, self.crop_w_start:self.crop_w_end, :]
        return cropped


class DuckieClipWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)

    def replace_nearby_colors(self, image, target_rgb, new_rgb, threshold=2):
        """
        Replaces all pixels in the image that are within a given threshold of the target RGB value.

        Parameters:
        - image: numpy array of shape (H, W, 3)
        - target_rgb: tuple of (R, G, B) values to match
        - new_rgb: tuple of (R, G, B) values to replace with
        - threshold: maximum difference for each channel to be considered a match

        Returns:
        - Modified image with replaced colors
        """
        # Convert to NumPy arrays
        #target_rgb = np.array(target_rgb, dtype=np.uint8)
        new_rgb = np.array(new_rgb, dtype=np.uint8)

        # Find pixels within the threshold
        mask = np.all(np.abs(image - target_rgb) <= threshold, axis=-1)

        # Replace matching pixels
        image[mask] = new_rgb

        return image

    def observation(self, observation):
        # Example usage:
        image = observation["camera_seg"]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_rgb = (128, 64, 128)  # RGB value to find
        new_rgb =(128, 64, 128),  # RGB value to replace with

        modified_image = self.replace_nearby_colors(image, target_rgb, new_rgb, threshold=2)
        target_rgb = (50,234, 157)# RGB value to find
        new_rgb =(50,234, 157) # RGB value to replace with

        modified_image = self.replace_nearby_colors(modified_image, target_rgb, new_rgb, threshold=20)
        # Create mask where pixels match the target color
        mask_1 = np.all(modified_image == (128,64,128), axis=-1, keepdims=True)
        mask_2 = np.all(modified_image == (50,234,157), axis=-1, keepdims=True)
        mask = mask_1 + mask_2
        image = modified_image * mask.astype(modified_image.dtype)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for color in [(128,64,128), (50,234,157)]:
            mask |= np.all(image == color, axis=-1)  # Mark pixels that match any of the colors

        # Apply the mask: Keep only selected colors, set others to black
        filtered_image = np.zeros_like(image)  # Create a black image
        filtered_image[mask == 1] = image[mask == 1]  # Copy only the kept colors
        result = observation
        result["camera_seg"] = filtered_image
        #cv2.imshow("modified", result["camera_seg"])
        #cv2.waitKey(1)
        return result

