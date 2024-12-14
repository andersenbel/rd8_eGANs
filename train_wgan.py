from wgan import WGANGenerator as Generator, WGANDiscriminator as Discriminator
from data_loader import get_dataloader
from config import CONFIG
from plot_utils import save_loss_plot
import torch
import torch.optim as optim
import os

# Налаштування
cfg = CONFIG['wgan']
device = CONFIG['device']
dataloader = get_dataloader(CONFIG['batch_size'])

# Моделі
generator = Generator(CONFIG['latent_dim']).to(device)
discriminator = Discriminator().to(device)

# Оптимізатори
optimizer_G = getattr(optim, cfg['optimizer']['type'])(
    generator.parameters(), lr=cfg['optimizer']['lr']
)
optimizer_D = getattr(optim, cfg['optimizer']['type'])(
    discriminator.parameters(), lr=cfg['optimizer']['lr']
)

# Логування втрат
generator_losses = []
discriminator_losses = []

# Тренування
for epoch in range(CONFIG['epochs']):
    g_loss_epoch = 0
    d_loss_epoch = 0

    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        z = torch.randn((imgs.size(0), CONFIG['latent_dim'], 1, 1)).to(device)

        # Тренування дискримінатора
        optimizer_D.zero_grad()
        real_loss = -torch.mean(discriminator(real_imgs))
        fake_imgs = generator(z)
        fake_loss = torch.mean(discriminator(fake_imgs.detach()))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Кліпінг ваг
        for p in discriminator.parameters():
            p.data.clamp_(-cfg['discriminator']['weight_clipping'],
                          cfg['discriminator']['weight_clipping'])

        # Тренування генератора
        optimizer_G.zero_grad()
        g_loss = -torch.mean(discriminator(fake_imgs))
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
               f"saved_models/wgan_generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(),
               f"saved_models/wgan_discriminator_epoch_{epoch}.pth")

# Збереження графіка втрат
save_loss_plot(generator_losses, discriminator_losses,
               "saved_models/wgan_loss.png")

# Збереження остаточних моделей
torch.save(generator.state_dict(), "saved_models/wgan_generator.pth")
torch.save(discriminator.state_dict(), "saved_models/wgan_discriminator.pth")
