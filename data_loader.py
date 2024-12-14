import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MNIST(root='./data', train=True,
                    transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
