import matplotlib.pyplot as plt
import os


def save_loss_plot(generator_losses, discriminator_losses, filename):
    """Зберігає графік втрат генератора та дискримінатора."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
