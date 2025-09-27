from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from torch import nn, optim, stack, Tensor
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from eval.abc import Result, Runner
from gan.gan import ConditionalGANAugmentor
from utils.logger import get_logger

logger = get_logger(__name__)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class Flower102Runner(Runner):
    METRICS_DIR: Path = Path("logs/naive_flower102cnn")

    def __init__(self, batch_size: int = 32, num_classes: int = 102, z_dim: int = 100):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.z_dim = z_dim

    def get_dataloaders(
        self, augment: bool = False, augment_epochs: int = 5
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        # Add proper normalization for Flowers102
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = datasets.Flowers102(
            root="./dataset",
            split="train",
            download=True,
            transform=train_transform,
        )
        val_dataset = datasets.Flowers102(
            root="./dataset", split="val", download=True, transform=test_transform
        )
        test_dataset = datasets.Flowers102(
            root="./dataset", split="test", download=True, transform=test_transform
        )

        if augment:
            # Use transform without normalization for GAN training
            gan_transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
            gan_dataset = datasets.Flowers102(
                root="./dataset", split="train", download=True, transform=gan_transform
            )
            train_loader_tmp = DataLoader(
                gan_dataset, batch_size=self.batch_size, shuffle=True
            )

            cgan_augmentor = ConditionalGANAugmentor(
                train_loader_tmp,
                z_dim=self.z_dim,
                img_size=224,
                num_classes=self.num_classes,
                channels=3,
            )
            cgan_augmentor.train_gan(epochs=augment_epochs)
            synthetic_imgs, synthetic_labels = cgan_augmentor.generate_synthetic(
                n_samples=len(train_dataset), save_dir=(self.METRICS_DIR / "runs")
            )

            # Normalize synthetic images
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            synthetic_imgs = stack([normalize(img) for img in synthetic_imgs])

            synthetic_dataset = TensorDataset(synthetic_imgs, synthetic_labels)
            train_dataset = ConcatDataset([train_dataset, synthetic_dataset])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        return train_loader, val_loader, test_loader

    def train_model(
        self,
        model: nn.Module,
        epochs: int = 10,
        augment: bool = False,
        augment_epochs: int = 5,
    ) -> tuple[nn.Module, Result]:
        train_loader, val_loader, test_loader = self.get_dataloaders(
            augment=augment, augment_epochs=augment_epochs
        )
        model = model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        model = self.train(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            epochs=epochs,
            writer_log_dir=(self.METRICS_DIR / "runs"),
        ).to(self.device)

        metrics = self.test(model, test_loader)

        metrics_data = {
            "model_type": "SmallCNN",
            "epochs": epochs,
            "augment": augment,
            "augment_epochs": augment_epochs,
            "test_results": metrics.model_dump(),
        }

        timestamp = int(datetime.now().timestamp())
        with open(self.METRICS_DIR / f"metrics_{timestamp}.json", "w") as file:
            json.dump(metrics_data, file)

        return model, metrics


app = typer.Typer()


@app.command()
def run(
    epochs: int = 10,
    batch_size: int = 32,
    augment: bool = False,
    augment_epochs: int = 10,
):
    """
    Train and test SmallCNN on Flowers102 with optional CGAN augmentation.
    """
    runner = Flower102Runner(batch_size=batch_size)
    model = SmallCNN(num_classes=102)
    trained_model, metrics = runner.train_model(
        model, epochs=epochs, augment=augment, augment_epochs=augment_epochs
    )
    typer.echo(
        f"Training complete. Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_macro:.4f}"
    )


if __name__ == "__main__":
    app()
