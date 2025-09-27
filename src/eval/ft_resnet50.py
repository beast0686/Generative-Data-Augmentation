from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from torch import nn, optim, stack, Tensor
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import datasets, transforms, models

from eval.abc import Result, Runner
from gan.gan import ConditionalGANAugmentor
from utils.logger import get_logger
from utils.tensor_dataset import TensorDatasetWrapper

logger = get_logger(__name__)


class FineTunedResNet50(nn.Module):
    """
    Fine-tuned ResNet50 for custom classification tasks.
    
    Uses pretrained ResNet50 backbone with modified classifier head.
    """
    
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    def unfreeze_backbone(self, unfreeze_layers: int = 2):
        """
        Unfreeze the last n layers of the backbone for fine-tuning.
        
        Args:
            unfreeze_layers: Number of layer groups to unfreeze from the end
        """
        layers_to_unfreeze = []
        
        if unfreeze_layers >= 1:
            layers_to_unfreeze.append(self.backbone.layer4)
        if unfreeze_layers >= 2:
            layers_to_unfreeze.append(self.backbone.layer3)
        if unfreeze_layers >= 3:
            layers_to_unfreeze.append(self.backbone.layer2)
        if unfreeze_layers >= 4:
            layers_to_unfreeze.append(self.backbone.layer1)
            
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
                
        logger.info(f"Unfroze last {len(layers_to_unfreeze)} layer groups")


class Flower102ResNetRunner(Runner):
    METRICS_DIR: Path = Path("logs/resnet50_flower102")

    def __init__(self, batch_size: int = 32, num_classes: int = 102, z_dim: int = 100):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.z_dim = z_dim

    def get_dataloaders(
        self, augment: bool = False, augment_epochs: int = 5
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        # Enhanced data augmentation for ResNet50
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Standard validation/test transform
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.Flowers102(
            root="./dataset", split="train", download=True, transform=train_transform
        )
        val_dataset = datasets.Flowers102(
            root="./dataset", split="val", download=True, transform=test_transform
        )
        test_dataset = datasets.Flowers102(
            root="./dataset", split="test", download=True, transform=test_transform
        )

        if augment:
            # Use transform without normalization for GAN training
            gan_transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor()
            ])
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

            train_dataset = ConcatDataset([TensorDatasetWrapper(train_dataset), TensorDatasetWrapper(synthetic_dataset)])

        # Optimized DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
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
        two_stage_training: bool = True,
    ) -> tuple[nn.Module, Result]:
        """
        Train ResNet50 with optional two-stage training:
        1. Train classifier head only (frozen backbone)
        2. Fine-tune last layers with lower learning rate
        """
        train_loader, val_loader, test_loader = self.get_dataloaders(
            augment=augment, augment_epochs=augment_epochs
        )
        model = model.to(self.device)

        if two_stage_training and hasattr(model, 'unfreeze_backbone'):
            # Stage 1: Train classifier head only
            logger.info("Stage 1: Training classifier head with frozen backbone")
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=1e-3, 
                weight_decay=1e-4
            )
            loss_fn = nn.CrossEntropyLoss()
            
            stage1_epochs = max(1, epochs // 3)  # Use 1/3 of epochs for stage 1
            model = self.train(
                model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                epochs=stage1_epochs,
                writer_log_dir=(self.METRICS_DIR / "runs" / "stage1"),
            )
            
            # Stage 2: Fine-tune with unfrozen layers
            logger.info("Stage 2: Fine-tuning with unfrozen backbone layers")
            model.unfreeze_backbone(unfreeze_layers=2)  # Unfreeze last 2 layer groups
            
            # Lower learning rate for fine-tuning
            optimizer = optim.Adam([
                {'params': model.backbone.fc.parameters(), 'lr': 1e-3},  # Higher LR for new layers
                {'params': model.backbone.layer4.parameters(), 'lr': 1e-4},  # Lower LR for pretrained
                {'params': model.backbone.layer3.parameters(), 'lr': 1e-5},  # Even lower LR
            ], weight_decay=1e-4)
            
            stage2_epochs = epochs - stage1_epochs
            model = self.train(
                model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                epochs=stage2_epochs,
                writer_log_dir=(self.METRICS_DIR / "runs" / "stage2"),
            )
        else:
            # Single stage training
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss_fn = nn.CrossEntropyLoss()
            model = self.train(
                model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                epochs=epochs,
                writer_log_dir=(self.METRICS_DIR / "runs"),
            )

        # Final evaluation
        metrics = self.test(model, test_loader)

        # Save comprehensive metrics
        metrics_data = {
            "model_type": "FineTunedResNet50",
            "epochs": epochs,
            "augment": augment,
            "augment_epochs": augment_epochs,
            "two_stage_training": two_stage_training,
            "test_results": metrics.model_dump(),
        }

        timestamp = int(datetime.now().timestamp())
        with open(self.METRICS_DIR / f"metrics_{timestamp}.json", "w") as file:
            json.dump(metrics_data, file, indent=2)

        return model, metrics


app = typer.Typer()


@app.command()
def run(
    epochs: int = 15,
    batch_size: int = 32,
    augment: bool = False,
    augment_epochs: int = 10,
    freeze_backbone: bool = True,
    two_stage_training: bool = True,
):
    """
    Train and test FineTunedResNet50 on Flowers102 with optional CGAN augmentation.
    
    Args:
        epochs: Total training epochs
        batch_size: Batch size for training
        augment: Whether to use GAN augmentation
        augment_epochs: Epochs for GAN training
        freeze_backbone: Whether to freeze ResNet backbone initially
        two_stage_training: Use two-stage training (recommended)
    """
    runner = Flower102ResNetRunner(batch_size=batch_size)
    model = FineTunedResNet50(num_classes=102, freeze_backbone=freeze_backbone)
    
    trained_model, metrics = runner.train_model(
        model, 
        epochs=epochs, 
        augment=augment, 
        augment_epochs=augment_epochs,
        two_stage_training=two_stage_training
    )
    
    typer.echo(
        f"Training complete. Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_macro:.4f}"
    )


if __name__ == "__main__":
    app()
