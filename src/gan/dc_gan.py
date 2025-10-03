from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------
# Generator
# ------------------------
class ConvGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64, img_size=224, num_classes=102):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.init_size = img_size // 8
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, feature_maps * 8 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(feature_maps * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_maps * 8, feature_maps * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_maps * 4, feature_maps * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_maps * 2, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ------------------------
# Discriminator
# ------------------------
class ConvDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, feature_maps, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        conv_out_size = feature_maps * 4 * (img_size // 8) * (img_size // 8)
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.conv(img)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ------------------------
# Augmentor / Trainer
# ------------------------
class Augmentor:
    def __init__(self, train_loader, latent_dim=100, img_channels=3, feature_maps=64, img_size=224, num_classes=102):
        self.train_loader = train_loader
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = ConvGenerator(latent_dim, img_channels, feature_maps, img_size, num_classes).to(self.device)
        self.discriminator = ConvDiscriminator(img_channels, feature_maps, img_size).to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(64, latent_dim, device=self.device)

    def train(self, epochs=20, label_smoothing=True):
        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(self.train_loader):
                batch_size = imgs.size(0)
                real_imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)
                if label_smoothing:
                    valid = valid * 0.9

                # Train Generator
                self.optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                random_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                gen_imgs = self.generator(z, random_labels)
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = self.criterion(self.discriminator(real_imgs), valid)
                fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

            logger.info(f"Epoch [{epoch + 1}/{epochs}] | Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    def generate_synthetic(self, n_samples, batch_size=64, save_dir: str | Path = "logs"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.generator.eval()
        all_generated = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                curr_batch = min(batch_size, n_samples - i)
                z = torch.randn(curr_batch, self.latent_dim, device=self.device)
                labels = torch.randint(0, self.num_classes, (curr_batch,), device=self.device)
                gen_imgs = self.generator(z, labels).cpu()

                for j in range(gen_imgs.shape[0]):
                    img_idx = i + j
                    save_image(gen_imgs[j], save_dir / f"synthetic_{img_idx:04d}.png", normalize=True)

                all_generated.append(gen_imgs)
                all_labels.append(labels.cpu())

        return torch.cat(all_generated, dim=0), torch.cat(all_labels, dim=0)