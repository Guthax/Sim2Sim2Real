import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from encoder import VariationalEncoder
from decoder import Decoder


# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 95


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims=LATENT_SPACE):
        super(VariationalAutoencoder, self).__init__()
        self.model_file = os.path.join('autoencoder/model', 'var_autoencoder.pth')
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
    
    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()
