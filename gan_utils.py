import matplotlib.pyplot as plt
import torchvision.utils as vutils


def save_generated_images(generator, epoch, path="output"):
    import os
    os.makedirs(path, exist_ok=True)

    z = torch.randn(16, 100, 1, 1).to(generator.device)
    with torch.no_grad():
        gen_imgs = generator(z).cpu()
    grid = vutils.make_grid(gen_imgs, nrow=4, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f"{path}/epoch_{epoch}.png")
    plt.close()
