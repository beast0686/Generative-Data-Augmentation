import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 224, 224]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 16, 112, 112]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 112, 112]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 32, 56, 56]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 56, 56]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 64, 28, 28]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
