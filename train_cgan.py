from cgan import CGANGenerator as Generator, CGANDiscriminator as Discriminator
from data_loader import get_dataloader
from config import CONFIG
from plot_utils import save_loss_plot
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Налаштування
cfg = CONFIG['cgan']
device = CONFIG['device']
dataloader = get_dataloader(CONFIG['batch_size'])

# Додавання перевірки ключів у CONFIG
latent_dim = CONFIG.get('latent_dim', 100)
num_classes = CONFIG.get('num_classes', 10)
img_shape = CONFIG.get('img_shape', (1, 28, 28))

# Моделі
generator = Generator(latent_dim, num_classes, img_shape).to(device)
discriminator = Discriminator(num_classes, img_shape).to(device)

# Оптимізатори
optimizer_G = getattr(optim, cfg['optimizer']['type'])(
    generator.parameters(), lr=cfg['optimizer']['lr'], betas=cfg['optimizer']['betas']
)
optimizer_D = getattr(optim, cfg['optimizer']['type'])(
    discriminator.parameters(), lr=cfg['optimizer']['lr'], betas=cfg['optimizer']['betas']
)

# Функція втрат
loss_fn = nn.BCELoss()

# Директорія для збереження моделей
os.makedirs("saved_models", exist_ok=True)

# Логування втрат
generator_losses = []
discriminator_losses = []

# Тренування
for epoch in range(CONFIG['epochs']):
    g_loss_epoch = 0
    d_loss_epoch = 0

    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        valid = torch.ones((imgs.size(0), 1), device=device)
        fake = torch.zeros((imgs.size(0), 1), device=device)
        noise = torch.randn((imgs.size(0), latent_dim), device=device)

        # Тренування дискримінатора
        optimizer_D.zero_grad()
        real_loss = loss_fn(discriminator(real_imgs, labels), valid)
        fake_imgs = generator(noise, labels)

        # Діагностика розмірів
        # print(f"Generated fake images shape: {fake_imgs.shape}")

        fake_loss = loss_fn(discriminator(fake_imgs.detach(), labels), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Тренування генератора
        optimizer_G.zero_grad()
        g_loss = loss_fn(discriminator(fake_imgs, labels), valid)
        g_loss.backward()
        optimizer_G.step()

        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()

    # Логування середніх втрат за епоху
    generator_losses.append(g_loss_epoch / len(dataloader))
    discriminator_losses.append(d_loss_epoch / len(dataloader))

    print(f"[Epoch {epoch}/{CONFIG['epochs']}] [D loss: {d_loss_epoch /
          len(dataloader)}] [G loss: {g_loss_epoch / len(dataloader)}]")

    # Збереження моделей після кожної епохи
    torch.save(generator.state_dict(),
               f"saved_models/cgan_generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(),
               f"saved_models/cgan_discriminator_epoch_{epoch}.pth")

# Збереження графіка втрат
save_loss_plot(generator_losses, discriminator_losses,
               "saved_models/cgan_loss.png")

# Збереження остаточних моделей
torch.save(generator.state_dict(), "saved_models/cgan_generator.pth")
torch.save(discriminator.state_dict(), "saved_models/cgan_discriminator.pth")
