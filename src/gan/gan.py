from pathlib import Path

import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.logger import get_logger

logger = get_logger(__name__)

HIDDEN_DIM_1: int = 256
HIDDEN_DIM_2: int = 512
HIDDEN_DIM_3: int = 1024
BATCH_SIZE: int = 32
LR: float = 2e-4
BETA1: float = 0.5
BETA2: float = 0.999


class CGANGenerator(nn.Module):
    """
    Conditional GAN Generator.

    Generates images conditioned on class labels.

    Formula for forward pass:
    .. math::
        \\text{img} = \\text{Tanh}(W_3 \\cdot \\text{ReLU}(W_2 \\cdot \\text{ReLU}(W_1 \\cdot [z, y]))))

    Parameters
    ----------
    z_dim : int
        Dimension of the latent vector z.
    num_classes : int
        Number of distinct class labels.
    img_size : int
        Height and width of output images (square assumed).
    channels : int
        Number of image channels (e.g., 3 for RGB).
    """

    def __init__(self, z_dim: int, num_classes: int, img_size: int, channels: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, HIDDEN_DIM_1),
            nn.ReLU(True),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(True),
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3),
            nn.ReLU(True),
            nn.Linear(HIDDEN_DIM_3, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        z : Tensor
            Latent noise vector of shape (batch_size, z_dim)
        labels : Tensor
            Class labels tensor of shape (batch_size,)

        Returns
        -------
        Tensor
            Generated images tensor of shape (batch_size, channels, img_size, img_size)
        """
        label_emb = self.label_emb(labels)
        x = torch.cat([z, label_emb], dim=1)
        img = self.model(x)
        img = img.view(z.size(0), self.channels, self.img_size, self.img_size)
        return img


class CGANDiscriminator(nn.Module):
    """
    Conditional GAN Discriminator.

    Estimates probability that an image is real, conditioned on the label.

    Formula for loss:
    .. math::
        L_D = - \\frac{1}{N} \\sum_{i=1}^N [y_i \\log D(x_i, c_i) + (1-y_i) \\log (1 - D(G(z_i, c_i)))]
    """

    def __init__(self, num_classes: int, img_size: int, channels: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels

        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = channels * img_size * img_size + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM_1, 1),
            nn.Sigmoid()
        )
    def forward(self, img: Tensor, labels: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        img : Tensor
            Input images (batch_size, channels, img_size, img_size)
        labels : Tensor
            Class labels (batch_size,)

        Returns
        -------
        Tensor
            Probability that image is real
        """
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_emb(labels)
        x = torch.cat([img_flat, label_emb], dim=1)
        validity = self.model(x)
        return validity


class ConditionalGANAugmentor:
    """
    Wrapper for training and generating synthetic images with CGAN.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader for training dataset
    z_dim : int
        Latent vector dimension
    img_size : int
        Output image size
    num_classes : int
        Number of class labels
    device : torch.device | None
        Torch device to use
    """

    def __init__(
        self,
        dataloader: DataLoader,
        z_dim: int,
        img_size: int,
        num_classes: int,
        channels: int,
        device: torch.device | None = None
    ) -> None:
        self.dataloader = dataloader
        self.z_dim = z_dim
        self.img_size = img_size
        self.num_classes = num_classes
        self.channels = channels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = CGANGenerator(z_dim, num_classes, img_size, channels).to(self.device)
        self.discriminator = CGANDiscriminator(num_classes, img_size, channels).to(self.device)

        self.optim_G = optim.Adam(self.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
        self.adversarial_loss = nn.BCELoss()

    def train_gan(self, epochs: int) -> None:
        """
        Train the CGAN model.

        Parameters
        ----------
        epochs : int
            Number of training epochs
        """
        logger.info("Starting CGAN training...")
        for epoch in range(epochs):
            d_loss_total, g_loss_total, batch_count = 0.0, 0.0, 0

            for imgs, labels in self.dataloader:
                batch_size = imgs.size(0)
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                valid = torch.ones(batch_size, 1, device=self.device) * 0.9
                fake = torch.zeros(batch_size, 1, device=self.device) + 0.1

                # Train Generator
                self.optim_G.zero_grad()
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                gen_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                gen_imgs = self.generator(z, gen_labels)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs, gen_labels), valid)
                g_loss.backward()
                self.optim_G.step()

                # Train Discriminator
                self.optim_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), gen_labels), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optim_D.step()

                d_loss_total += d_loss.item()
                g_loss_total += g_loss.item()
                batch_count += 1

            avg_d_loss = d_loss_total / batch_count if batch_count > 0 else 0.0
            avg_g_loss = g_loss_total / batch_count if batch_count > 0 else 0.0
            logger.info(f"Epoch {epoch+1}/{epochs} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    def generate_synthetic(self, n_samples: int, save_dir: str | None = None) -> tuple[Tensor, Tensor]:
        """
        Generate synthetic images and optionally save them.

        Parameters
        ----------
        n_samples : int
            Number of images to generate
        save_dir : str | None
            Directory to save images (optional)

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple of (images, labels)
        """
        self.generator.eval()
        imgs: list[Tensor] = []
        labels_list: list[Tensor] = []

        save_path: Path | None = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for _ in range((n_samples + BATCH_SIZE - 1) // BATCH_SIZE):
                batch_size = min(BATCH_SIZE, n_samples - len(imgs) * BATCH_SIZE)
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                batch_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                gen_imgs = self.generator(z, batch_labels)

                if save_path:
                    for i, img in enumerate(gen_imgs):
                        filename = save_path / f"generated_{len(imgs)*BATCH_SIZE + i}.png"
                        save_image(img, filename, normalize=True)

                imgs.append(gen_imgs.cpu())
                labels_list.append(batch_labels.cpu())

        imgs_tensor = torch.cat(imgs, dim=0)
        labels_tensor = torch.cat(labels_list, dim=0)
        return imgs_tensor, labels_tensor
