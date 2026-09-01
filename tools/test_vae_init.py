import torch
import sys
from pathlib import Path

# Import from script without .py extension
exec(open("02_vae_model.py").read())

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

model = VAE(in_channels=2, latent_dim=64, h=91, w=180)
print(f"VAE model instantiated: {model}")

x_dummy = torch.randn(2, 2, 91, 180)
with torch.no_grad():
    x_recon, mu, logvar = model(x_dummy)
    print(f"Input shape: {x_dummy.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}, Logvar shape: {logvar.shape}")
    print("Model forward pass OK")
