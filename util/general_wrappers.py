from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperObsType
from stable_baselines3.common.utils import obs_as_tensor
from torchvision.transforms import transforms
from vae.encoder import VariationalEncoder
import matplotlib.pyplot as plt


class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        if "camera_seg" in self.observation_space.spaces:
            c, h, w =  self.observation_space["camera_seg"].shape[2], self.observation_space["camera_seg"].shape[0], self.observation_space["camera_seg"].shape[1]
            self.observation_space["camera_seg"] = gym.spaces.Box(low=0, high=1, shape=(c, h, w), dtype=np.uint8)
        if "camera_rgb" in self.observation_space.spaces:
            c, h, w = self.observation_space["camera_rgb"].shape[2], self.observation_space["camera_rgb"].shape[0], self.observation_space["camera_rgb"].shape[1]
            self.observation_space["camera_rgb"] = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def observation(self, observation):
        if "camera_seg" in observation:
            observation["camera_seg"] = np.transpose(observation["camera_seg"], (2, 0, 1))  # (H, W, C) -> (C, H, W)
        if "camera_rgb" in observation:
            observation["camera_rgb"] = np.transpose(observation["camera_rgb"], (2, 0, 1))
        return observation

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        # Assumes channel first
        super().__init__(env)
        if "camera_rgb" in self.observation_space.spaces:
            c = self.observation_space["camera_rgb"].shape[0]
            self.observation_space["camera_rgb"] = gym.spaces.Box(low=0, high=1, shape=(c, 120, 160), dtype=np.float32)

    def observation(self, observation):
        if "camera_rgb" in observation:
            observation["camera_rgb"] = observation["camera_rgb"] / 255
        return observation


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

class OneHotEncodeSegWrapper(gym.ObservationWrapper):
    color_map = {
        (0, 0, 0): 0,  # Background
        (128, 64, 128): 1,  # Class 1 (Road)
        (157, 234, 50): 2,  # Class 2 (Markigns)
    }

    reset_every = 10000

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)

        h, w = self.observation_space["camera_seg"].shape[0], self.observation_space["camera_seg"] .shape[1]
        self.observation_space["camera_seg"] = spaces.Box(low=0, high=1, shape=(h, w, len(self.color_map)), dtype=np.uint8)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def observation(self, observation):
        array = observation["camera_seg"]
        one_hot_mask = self.one_hot_encode_segmentation_torch(torch.from_numpy(array).to(torch.uint8))
        one_hot_mask[:, :, 0] = 0
        observation["camera_seg"] = one_hot_mask
        return observation


    def one_hot_encode_segmentation_torch(self,mask, background_class=0):
        """
        Convert an HxWx3 RGB segmentation mask to a CxHxW one-hot encoded representation using PyTorch.
        Unrecognized colors are assigned to the background class.

        Args:
            mask (torch.Tensor): HxWx3 segmentation mask (RGB format, dtype=torch.uint8).
            color_map (dict): Dictionary mapping RGB tuples to class indices.
            background_class (int): Class index for background.
            device (str): Device to run on ("cpu" or "cuda").

        Returns:
            torch.Tensor: One-hot encoded mask of shape (C, H, W) on the specified device.
        """
        color_map = self.color_map
        device = self.device

        mask = mask.to(device)  # Move the mask to the desired device (CPU or GPU)
        H, W, _ = mask.shape
        C = len(color_map)  # Number of classes to map (excluding background)

        # Convert color_map to tensor for efficient processing
        colors = torch.tensor(list(color_map.keys()), dtype=torch.uint8, device=device)  # (num_classes, 3)
        class_indices = torch.tensor(list(color_map.values()), dtype=torch.long, device=device)  # (num_classes,)

        # Flatten the mask for faster processing
        mask_flat = mask.reshape(-1, 3)  # (H*W, 3)

        # Find matching colors using broadcasting (vectorized comparison)
        matches = (mask_flat[:, None, :] == colors).all(dim=2)  # (H*W, num_classes), True where match found

        # Convert matches to int (to use argmax), and then find the class index
        matches_int = matches.to(torch.long)
        mapped_indices = torch.where(
            matches_int.any(dim=1),
            class_indices[matches_int.argmax(dim=1)],  # Get the class index of the first match
            torch.tensor(background_class, device=device)  # Default to background class
        )

        # Reshape to (H, W)
        class_mask = mapped_indices.view(H, W)

        # One-hot encode the class indices
        one_hot_mask = torch.nn.functional.one_hot(class_mask, num_classes=C).to(
            torch.float32).cpu().numpy()

        return one_hot_mask

class EncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.encoder = VariationalEncoder()
        self.encoder.load()
        self.encoder.to(device)

        self.observation_space = gym.spaces.Dict({
            "encoding": gym.spaces.Box(low=-10, high=10, shape=(95,), dtype=np.float32),
            "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1))
        })

    def observation(self, observation):
        img = observation["camera_seg"]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        encoding = self.encoder.forward(tensor).cpu().numpy()

        obs = {
            "encoding": encoding,
            "vehicle_dynamics": observation["vehicle_dynamics"]
        }
        return obs

class CannyWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
        )

    def observation(self, observation):
        processed_image = self.Lane_Mask_Pipeline(observation)
        processed_image[:40, :] = 0
        processed_image = np.expand_dims(processed_image, axis=2)
        cv2.imshow("processed", processed_image)
        cv2.waitKey(1)
        return processed_image

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
        if len(img.shape) > 2:
            ignore_mask_color = (255,) * img.shape[2]
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)

    @staticmethod
    def extrapolate(x, y):
        if len(x) == 0 or len(y) == 0:
            return None
        z = np.polyfit(x, y, 1)
        f = np.poly1d(z)
        x_new = np.linspace(min(x), max(x), 10).astype(int)
        y_new = f(x_new).astype(int)
        return x_new[0], y_new[0], x_new[-1], y_new[-1]

    @classmethod
    def draw_lines(cls, img, lines, color=[255, 255, 255], thickness=2):
        xp, yp, xn, yn = [], [], [], []

        if lines is None:
            return

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope > 0.5:
                    xp.extend([x1, x2])
                    yp.extend([y1, y2])
                elif slope < -0.5:
                    xn.extend([x1, x2])
                    yn.extend([y1, y2])

        if xp and yp:
            result = cls.extrapolate(xp, yp)
            if result:
                x1, y1, x2, y2 = result
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        if xn and yn:
            result = cls.extrapolate(xn, yn)
            if result:
                x1, y1, x2, y2 = result
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    @classmethod
    def hough_lines(cls, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((*img.shape, 1), dtype=np.uint8)
        cls.draw_lines(line_img, lines)
        return line_img

    @classmethod
    def Lane_Mask_Pipeline(cls, image):
        # Apply grayscale, Gaussian blur, and Canny edge detection
        gray = cls.grayscale(image)
        blur_gray = cls.gaussian_blur(gray, 5)

        # Lowering the thresholds to detect more edges
        edges = cls.canny(blur_gray, low_threshold=30, high_threshold=100)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        line_image = np.zeros_like(edges)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

        return line_image


class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, keys=[], crop_height_start= 40, crop_height_end=120, crop_width_start=0, crop_width_end=160, channels=3):
        gym.ObservationWrapper.__init__(self, env)

        self.crop_h_start =crop_height_start
        self.crop_h_end = crop_height_end
        self.crop_w_start = crop_width_start
        self.crop_w_end = crop_width_end
        self.num_channels = channels
        self.keys = keys
        for key in self.keys:
            if key in self.observation_space.spaces:
                self.observation_space[key] =  spaces.Box(low=0, high=255, shape=(crop_height_end - crop_height_start, crop_width_end - crop_width_start, channels), dtype=np.uint8)

    def observation(self, observation):

        for key in self.keys:
            if key in observation:
                observation[key] = observation[key][self.crop_h_start:self.crop_h_end, self.crop_w_start:self.crop_w_end, :]
        return observation

class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.observation_space.shape[0], self.observation_space.shape[1], 1), dtype=np.uint8)

    def observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=2)  # Expand dimensions for compatibility
        cv2.imshow("gray", gray)
        cv2.waitKey(1)
        return gray


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
        image = observation
        if isinstance(self.env.observation_space, spaces.Dict):
            image = observation["camera_seg"]

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_rgb = (128, 64, 128)  # RGB value to find
        new_rgb =(128, 64, 128),  # RGB value to replace with

        modified_image = self.replace_nearby_colors(image, target_rgb, new_rgb, threshold=8)
        target_rgb = (157,234, 50)# RGB value to find
        new_rgb =(157,234, 50)# RGB value to replace with

        modified_image = self.replace_nearby_colors(modified_image, target_rgb, new_rgb, threshold=20)
        if isinstance(self.env.observation_space, spaces.Dict):
            observation["camera_seg"] = modified_image
            return observation

        return modified_image