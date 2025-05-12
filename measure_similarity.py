import math
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
    final_representation_actor, extract_features, extract_features_per_layer
from util.similarity import cca, gram_linear, feature_space_linear_cka, cka
from utils import lr_schedule
from skimage.metrics import structural_similarity as ssim, mean_squared_error

import torch.nn.functional as F


class CKA:
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

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

        features_per_layer_per_model = [{} for i in range(len(self.models))]
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if key =="camera_rgb":
                frame_processed = np.expand_dims(np.transpose(frame[:][40:], (2,0,1)) / 255.0, axis=0)
                cv2.imshow("Test",frame[:][40:])
                cv2.waitKey(1)
            elif key =="camera_gray":
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[:][40:]
                frame_processed = np.expand_dims(np.expand_dims(frame_processed, axis=0), axis=0)
            else:
                frame_processed = np.expand_dims(np.expand_dims(frame[:][40:] / 255.0, axis=0), axis=0)

            obs = {
                key: deepcopy(frame_processed),
                "vehicle_dynamics": [[0]],
            }

            for i, model in enumerate(self.models):
                fr = final_representation(model, obs, key).detach().cpu().numpy()
                features = extract_features(model, obs)

                features_per_layer = extract_features_per_layer(model, obs)

                for k, v in features_per_layer.items():
                    if k in features_per_layer_per_model[i]:
                        features_per_layer_per_model[i][k] = np.concatenate(
                            (features_per_layer_per_model[i][k], np.expand_dims(v, axis=0)), axis=0
                        )
                    else:
                        features_per_layer_per_model[i][k] = np.expand_dims(v, axis=0)
                samples[i].append(fr)
                feats[i].append(features.detach().cpu().numpy().flatten())

            #fr_a1 = final_representation_actor(self.model_1, obs).detach().cpu().numpy()
            #fr_a2 = final_representation_actor(self.model_2, obs).detach().cpu().numpy()
            #samples_action_m1.append(fr_a1[0])
            #samples_action_m2.append(fr_a2[0])
        """ 
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
        """
        samples_m1 = np.array(samples[0]).squeeze(1)
        samples_m2 = np.array(samples[1]).squeeze(1)


        np_cka = CKA()

        print('Linear CKA, between X and Y: {}'.format(np_cka.linear_CKA(samples_m1, samples_m2)))
        print('Linear CKA, between Y and X: {}'.format(np_cka.linear_CKA(samples_m2, samples_m1)))


        data = np.zeros((len(features_per_layer_per_model[0].keys()),len(features_per_layer_per_model[1].keys())))
        x_labels = features_per_layer_per_model[0].keys()
        y_labels = features_per_layer_per_model[1].keys()
        for i, key in enumerate(features_per_layer_per_model[0].keys()):
            for i2, key2 in enumerate(features_per_layer_per_model[1].keys()):
                samples_m1 = features_per_layer_per_model[0][key]
                samples_m2 = features_per_layer_per_model[1][key2]


                cca_metric = np_cka.linear_CKA(samples_m1, samples_m2)

                data[i][i2] = cca_metric
        plt.imshow(data, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(ticks=np.arange(len(features_per_layer_per_model[0].keys())), labels=x_labels, rotation=45, ha='right', fontsize=10)
        plt.yticks(ticks=np.arange(len(features_per_layer_per_model[1].keys())), labels=y_labels, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center',
                         color='white' if data[i, j] < 0.5 else 'black')

        plt.title("Heatmap of layer similarity between Gray+No  DR and Baseline")
        plt.tight_layout()
        plt.show()

        print("done")



model_1 = PPO.load("/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/256/carla_gray_256_no_dr_crop_model_trained_1000000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
model_2 = PPO.load("/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/256/duckie_gray_256_model_trained_400000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
#model_3 = PPO.load("/home/jurriaan/workplace/programming/Sim2Sim2Real/results/carla_rgb_256_model_trained_400000_steps", device='cuda' if torch.cuda.is_available() else 'cpu')
ge = SimilarityMeasurer(video_path='/home/jurriaan/Documents/Programming/Sim2Sim2Real/test/videos/duckiebot_real.mp4',
                      models = [model_1, model_2])
ge.run(key="camera_gray")
