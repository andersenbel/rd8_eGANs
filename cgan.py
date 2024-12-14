from config import CONFIG
import torch.nn as nn


import torch
import torch.nn as nn


class CGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super(CGANGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.img_shape = img_shape
        input_dim = latent_dim + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embeddings = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_embeddings), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class CGANDiscriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super(CGANDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.img_shape = img_shape
        input_dim = int(torch.prod(torch.tensor(img_shape))) + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embeddings = self.label_embedding(labels)
        flat_img = img.view(img.size(0), -1)
        d_input = torch.cat((flat_img, label_embeddings), -1)
        validity = self.model(d_input)
        return validity
