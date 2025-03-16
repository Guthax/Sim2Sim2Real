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
        gym.ObservationWrapper.__init__(self, env)
        self.resize_h = dst_height
        self.resize_w = dst_width
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.resize_h, self.resize_w, 3), dtype=np.uint8)

    def observation(self, observation):
        img = observation
        obs =  cv2.resize(img, dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)
        return obs


class SegmentationFilterWrapper(gym.ObservationWrapper):
    colors_to_keep_seg = [
        (128, 64, 128),
        (157, 234, 50)
    ]


    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self.gray_value = np.random.randint(0, 256, dtype=np.uint8)
        #window = cv2.namedWindow("filtered")

    def reset(self, **kwargs):
        """Reset environment and randomize gray background."""
        self.gray_value = np.random.randint(0, 256, dtype=np.uint8)  # Generate gray value on reset
        return super().reset(**kwargs)

    def observation(self, observation):
        array = observation
        mask = np.zeros(array.shape[:2], dtype=np.uint8)
        for color in self.colors_to_keep_seg:
            mask |= np.all(array == color, axis=-1)  # Mark pixels that match any of the colors

        # Apply the mask: Keep only selected colors, set others to black
        filtered_image = np.full_like(array, self.gray_value)  # Create a black image
        filtered_image[mask == 1] = array[mask == 1]  # Copy only the kept colors
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("filtered", filtered_image)
        #cv2.waitKey(1)
        return filtered_image

class LaneMarkingWrapper(gym.ObservationWrapper):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(640, 640, 3), dtype=np.uint8)

        self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        self.model.to('cuda')
        self.model.eval()

    def observation(self, observation):
        img_rgb = observation
        img = self.transform(img_rgb).to(torch.device("cuda"))
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        det_out, da_seg_out, ll_seg_out = self.model(img)
        _, _, height, width = img.shape
        b, h, w, _ = img.shape
        pad_w, pad_h = 0, 0
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = 1

        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        img_det = self.show_seg_result(img_rgb, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        return img_det

    def show_seg_result(self, img, result, save_dir=None, is_ll=True, palette=None, is_demo=False, is_gt=False):
        # img = mmcv.imread(img)
        # img = img.copy()
        # seg = result[0]
        image = img.copy()
        if palette is None:
            palette = np.random.randint(
                0, 255, size=(3, 3))
        palette[0] = [0, 0, 0]
        palette[1] = [0, 255, 0]
        palette[2] = [255, 0, 0]
        palette = np.array(palette)
        assert palette.shape[0] == 3  # len(classes)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        if not is_demo:
            color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[result == label, :] = color
        else:
            color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

            # for label, color in enumerate(palette):
            #     color_area[result[0] == label, :] = color

            #color_area[result[0] == 1] = [0, 255, 0]
            color_area[result[1] == 1] = [255, 0, 0]
            color_seg = color_area

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        # print(color_seg.shape)
        color_mask = np.mean(color_seg, 2)
        image[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        # img = img * 0.5 + color_seg * 0.5
        image = image.astype(np.uint8)
        #img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        return color_seg

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