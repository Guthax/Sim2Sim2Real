import torch

from vae.models import VAE
from vae_plots import VAEVisualizer

model = VAE(64).to(torch.device("cuda"))
state = torch.load("F:\Github\Sim2Sim2Real\\vae\\log_dir\\vae_64\checkpoint.tar")

print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
model.load_state_dict(state['state_dict'])

model.eval()

# Initialize the VAE visualizer
image_path = 'F:\Github\Sim2Sim2Real\\vae\original.jpg'  # path to the sample image

image_path_list = [image_path]
visualizer = VAEVisualizer(model, torch.device("cuda"))
visualizer.visualize(image_path)