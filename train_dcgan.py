from dcgan import Generator, Discriminator
from data_loader import get_dataloader
from config import CONFIG
from plot_utils import save_loss_plot
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Налаштування
cfg = CONFIG['dcgan']
device = CONFIG['device']
dataloader = get_dataloader(CONFIG['batch_size'])

# Моделі
generator = Generator(CONFIG['latent_dim']).to(device)
discriminator = Discriminator().to(device)

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

    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        valid = torch.ones((imgs.size(0), 1)).to(device)
        fake = torch.zeros((imgs.size(0), 1)).to(device)

        # Тренування дискримінатора
        optimizer_D.zero_grad()
        real_loss = loss_fn(discriminator(real_imgs), valid)
        z = torch.randn((imgs.size(0), CONFIG['latent_dim'], 1, 1)).to(device)
        fake_imgs = generator(z)

        # Діагностика розмірів
        # print(f"Generated fake images shape: {fake_imgs.shape}")

        fake_loss = loss_fn(discriminator(fake_imgs.detach()), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Тренування генератора
        optimizer_G.zero_grad()
        g_loss = loss_fn(discriminator(fake_imgs), valid)
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
               f"saved_models/dcgan_generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(),
               f"saved_models/dcgan_discriminator_epoch_{epoch}.pth")

# Збереження графіка втрат
save_loss_plot(generator_losses, discriminator_losses,
               "saved_models/dcgan_loss.png")

# Збереження остаточних моделей
torch.save(generator.state_dict(), "saved_models/dcgan_generator.pth")
torch.save(discriminator.state_dict(), "saved_models/dcgan_discriminator.pth")
