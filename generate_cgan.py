from cgan import CGANGenerator as Generator
import torch
import matplotlib.pyplot as plt
import os

# Параметри
latent_dim = 100
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження моделі
generator = Generator(latent_dim, num_classes, (1, 28, 28)).to(device)
generator.load_state_dict(torch.load(
    "saved_models/cgan_generator.pth", map_location=device))
generator.eval()

# Генерація зображень для класу 1
os.makedirs("generated_images", exist_ok=True)
labels = torch.tensor([1] * 16, device=device)
noise = torch.randn(16, latent_dim, device=device)
fake_imgs = generator(noise, labels).detach().cpu()

# Збереження зображень
for i, img in enumerate(fake_imgs):
    plt.imsave(
        f"generated_images/cgan_image_class1_{i}.png", img[0], cmap="gray")

print("Зображення для CGAN збережені в папку 'generated_images'")
