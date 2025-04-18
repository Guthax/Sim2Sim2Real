import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class GradCamEvaluator:
    def __init__(self, model_1_path=None, model_2_path=None, video_path=None):
        self.model_1 = PPO.load(model_1_path) if model_1_path else PPO("CnnPolicy", make_vec_env("CartPole-v1"),
                                                                       verbose=0)
        self.model_2 = PPO.load(model_2_path) if model_2_path else PPO("CnnPolicy", make_vec_env("CartPole-v1"),
                                                                       verbose=0)

        self.video_path = video_path
        self.test_video = None

    def _preprocess_frame(self, frame):
        # Resize and normalize to fit model expectations (e.g., 84x84 grayscale)
        processed = cv2.resize(frame, (84, 84))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=-1)  # Add channel
        processed = np.expand_dims(processed, axis=0)   # Add batch
        return processed

    def run(self):
        if not self.video_path:
            raise ValueError("Video path not provided.")

        cap = cv2.VideoCapture(self.video_path)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow(frame)
            cv2.waitKey(1)
