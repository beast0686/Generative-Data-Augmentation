import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Generator
# ----------------------------
class CGANGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=102, img_size=224, channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        img = img.view(z.size(0), self.channels, self.img_size, self.img_size)
        return img

# ----------------------------
# Discriminator
# ----------------------------
class CGANDiscriminator(nn.Module):
    def __init__(self, num_classes=102, img_size=224, channels=3):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.img_size = img_size
        self.channels = channels

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([img_flat, c], dim=1)
        validity = self.model(x)
        return validity

# ----------------------------
# CGAN augmentor
# ----------------------------
class ConditionalGANAugmentor:
    def __init__(self, dataloader: DataLoader, img_size=224, num_classes=102, z_dim=100, device=None):
        self.dataloader = dataloader
        self.img_size = img_size
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = CGANGenerator(z_dim=z_dim, num_classes=num_classes, img_size=img_size).to(self.device)
        self.discriminator = CGANDiscriminator(num_classes=num_classes, img_size=img_size).to(self.device)

        self.optim_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.adversarial_loss = nn.BCELoss()

    def train_gan(self, epochs=5):
        print("Training CGAN...")
        for epoch in range(epochs):
            d_loss_total = 0.0
            g_loss_total = 0.0
            batch_count = 0

            for imgs, labels in self.dataloader:
                batch_size = imgs.size(0)
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # Adversarial ground truths
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)

                # -----------------
                #  Train Generator
                # -----------------
                self.optim_G.zero_grad()
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                gen_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                gen_imgs = self.generator(z, gen_labels)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs, gen_labels), valid)
                g_loss.backward()
                self.optim_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
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
            print(f"Epoch {epoch+1}/{epochs} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    def generate_synthetic(self, n_samples=500):
        BATCH_SIZE = 32
        self.generator.eval()
        imgs, labels = [], []
        with torch.no_grad():
            for _ in range((n_samples + BATCH_SIZE - 1) // BATCH_SIZE):
                batch_size = min(BATCH_SIZE, n_samples - len(imgs))
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                batch_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                gen_imgs = self.generator(z, batch_labels)
                imgs.append(gen_imgs.cpu())
                labels.append(batch_labels.cpu())
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(labels, dim=0)
        return imgs, labels