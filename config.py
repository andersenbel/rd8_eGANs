CONFIG = {
    # Загальні параметри
    "latent_dim": 100,  # Розмір латентного простору (вхід генератора)
    "batch_size": 64,   # Розмір батчу
    "epochs": 10,       # Кількість епох
    "device": "cpu",    # Пристрій для тренування (cpu/gpu)

    # DCGAN
    "dcgan": {
        "generator": {
            "filters": [128, 64],  # Розміри фільтрів для кожного шару
            "activation": "ReLU",  # Активація (ReLU, LeakyReLU)
            "output_activation": "Tanh"  # Активація для вихідного шару
        },
        "discriminator": {
            "filters": [64, 128],  # Розміри фільтрів для кожного шару
            "activation": "LeakyReLU",  # Активація (ReLU, LeakyReLU)
            "normalization": True,  # Використовувати BatchNorm
            "dropout": 0.3          # Dropout для регуляризації
        },
        "loss": "binary_crossentropy",  # Функція втрат
        "optimizer": {
            "type": "Adam",         # Оптимізатор (Adam, RMSprop)
            "lr": 0.0002,           # Коефіцієнт навчання
            "betas": (0.5, 0.999)   # Параметри Adam
        }
    },

    # WGAN
    "wgan": {
        "generator": {
            "filters": [128, 64],
            "activation": "ReLU",
            "output_activation": "Tanh"
        },
        "discriminator": {
            "filters": [64, 128],
            "activation": "LeakyReLU",
            "normalization": False,
            "weight_clipping": 0.01  # Кліпінг ваг
        },
        "loss": "wasserstein",       # Функція втрат
        "optimizer": {
            "type": "RMSprop",
            "lr": 0.00005
        }
    },

    # cGAN
    "cgan": {
        "generator": {
            "filters": [128, 64],
            "activation": "ReLU",
            "output_activation": "Tanh",
            "n_classes": 10  # Кількість класів
        },
        "discriminator": {
            "filters": [64, 128],
            "activation": "LeakyReLU",
            "normalization": True
        },
        "loss": "binary_crossentropy",
        "optimizer": {
            "type": "Adam",
            "lr": 0.0002,
            "betas": (0.5, 0.999)
        }
    }
}
