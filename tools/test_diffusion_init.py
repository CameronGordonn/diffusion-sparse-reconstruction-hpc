exec(open("04_diffusion_model_minimal.py").read())
import torch

print("PyTorch version:", torch.__version__)
model = DiffusionUNet(in_channels=5, out_channels=2, base_channels=64)
print("Model instantiated OK")

x = torch.randn(2, 2, 91, 180)
t = torch.randint(0, 300, (2,))
y = torch.randn(2, 2, 91, 180)
M = torch.ones(2, 1, 91, 180)

with torch.no_grad():
    out = model(x, t, y, M)
    print(f"Forward pass OK. Output shape: {out.shape}")
print("Diffusion model test PASSED")
