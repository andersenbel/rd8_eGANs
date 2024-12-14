import torch
import torch.nn as nn


class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(WGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),  # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 2, 1),  # (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class WGANDiscriminator(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(WGANDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # (64, 14, 14)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.flatten_input_size = self._get_flatten_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_input_size, 1)  # Вихід без Sigmoid для WGAN
        )

    def _get_flatten_size(self, input_shape):
        """Обчислює розмірність після згорткових шарів."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self.conv_layers(dummy_input)
            return out.numel()

    def forward(self, img):
        out = self.conv_layers(img)
        out = out.view(out.size(0), -1)  # Розгортання у плоский вигляд
        return self.fc(out)
