from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import cv2
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor
from torch import cosine_similarity
from torch.backends.cudnn import deterministic

from util.grad_cam import grad_cam, feature_map, final_representation, compare_histograms, compare_image_histograms, \
    final_representation_actor, extract_features
from util.similarity import cca, gram_linear, feature_space_linear_cka, cka
from utils import lr_schedule
from skimage.metrics import structural_similarity as ssim, mean_squared_error

import torch.nn.functional as F



class SimilarityMeasurer:
    def __init__(self, models, video_path=None):

        self.video_path = video_path
        self.models = []
        for model in models:
            model.policy.eval()
            self.models.append(model)

    def run(self, key="camera_rgb"):
        if not self.video_path:
            raise ValueError("Video path not provided.")

        cap = cv2.VideoCapture(self.video_path)

        samples = [[] for i in range(len(self.models))]

        feats = [[] for i in range(len(self.models))]
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if key =="camera_rgb":
                frame_processed = np.expand_dims(np.transpose(frame[:][40:], (2,0,1)) / 255.0, axis=0)
            else:
                frame_processed = np.expand_dims(np.expand_dims(frame / 255.0, axis=0), axis=0)

            obs = {
                key: deepcopy(frame_processed),
                "vehicle_dynamics": [[0]],
            }

            for i, model in enumerate(self.models):
                fr = final_representation(model, obs, key).detach().cpu().numpy()
                features = extract_features(model, obs)

                samples[i].append(fr)
                feats[i].append(features.detach().cpu().numpy().flatten())


            #fr_a1 = final_representation_actor(self.model_1, obs).detach().cpu().numpy()
            #fr_a2 = final_representation_actor(self.model_2, obs).detach().cpu().numpy()
            #samples_action_m1.append(fr_a1[0])
            #samples_action_m2.append(fr_a2[0])

        for i in range(len(feats)):
            arr = np.array(feats[i])
            feats[i] = arr / np.linalg.norm(arr, axis=1, keepdims=True)

        # Stack features
        X = np.vstack(feats)

        # Create labels (0 for first model, 1 for second, 2 for third)
        y = np.concatenate([[i] * len(feats[i]) for i in range(len(feats))])

        # Choose dimensionality reduction method
        use_tsne = False  # Set to True for t-SNE, False for PCA

        if use_tsne:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:
            reducer = PCA(n_components=2)

        X_reduced = reducer.fit_transform(X)

        # Plot
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'orange', 'green']
        labels = ['CARLA', 'Duckietown', 'ThirdModel']

        for i in range(3):
            plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
                        label=labels[i], alpha=0.6, c=colors[i])

        plt.legend()
        plt.title(f"{'t-SNE' if use_tsne else 'PCA'} of PPO Latent Features Across Three Models")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        samples_m1 = np.array(samples[0]).squeeze(1)
        samples_m2 = np.array(samples[1]).squeeze(1)


        cca_score = cca(samples_m1, samples_m2)
        cka_from_examples = cka(gram_linear(samples_m1), gram_linear(samples_m2))
        cka_from_features = feature_space_linear_cka(samples_m1, samples_m2)

        print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
        print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
        print(f'CCA: {cca_score}')

        print("done")




model_1 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/duckie_rgb_256_model_trained_400000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
model_2 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/carla_rgb_256_model_trained_400000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
model_3 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/carla_rgb_256_model_trained_400000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
ge = SimilarityMeasurer(video_path='/home/jurriaan/workplace/programming/Sim2Sim2Real/test/videos/duckie_video_right_lane.mp4',
                      models = [model_1, model_2, model_3])
ge.run()
