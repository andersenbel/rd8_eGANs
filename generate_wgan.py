from wgan import WGANGenerator as Generator
import torch
import matplotlib.pyplot as plt
import os

# Параметри
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження моделі
generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load(
    "saved_models/wgan_generator.pth", map_location=device))
generator.eval()

# Генерація зображень
os.makedirs("generated_images", exist_ok=True)
noise = torch.randn(16, latent_dim, 1, 1, device=device)
fake_imgs = generator(noise).detach().cpu()

# Збереження зображень
for i, img in enumerate(fake_imgs):
    plt.imsave(f"generated_images/wgan_image_{i}.png", img[0], cmap="gray")

print("Зображення для WGAN збережені в папку 'generated_images'")
