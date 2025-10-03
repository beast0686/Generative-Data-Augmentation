import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# U-Net Components
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.up = up
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class UNet(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=32, base_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.conv0 = nn.Conv2d(img_channels, base_dim, 3, padding=1)
        self.down1 = Block(base_dim, base_dim, time_emb_dim)
        self.down2 = Block(base_dim, base_dim * 2, time_emb_dim)
        self.down3 = Block(base_dim * 2, base_dim * 2, time_emb_dim)

        # Bottleneck
        self.bot1 = nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1)
        self.bot2 = nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1)

        # Decoder
        # self.up1 = Block(base_dim*2, base_dim*2, time_emb_dim, up=True)
        # self.up2 = Block(base_dim*2, base_dim, time_emb_dim, up=True)
        # self.up3 = Block(base_dim, base_dim, time_emb_dim, up=True)
        # self.out = nn.Conv2d(base_dim*2, img_channels, 1)

        # Decoder
        self.up1 = Block(base_dim * 2, base_dim * 2, time_emb_dim, up=True)
        self.up2 = Block(base_dim * 4, base_dim, time_emb_dim, up=True)
        self.up3 = Block(base_dim * 2, base_dim, time_emb_dim, up=True)
        self.out = nn.Conv2d(base_dim * 2, img_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        # Encoder
        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)
        x3 = self.down3(x2, t)

        # Bottleneck
        x3 = F.relu(self.bot1(x3))
        x3 = F.relu(self.bot2(x3))

        # Decoder with skip connections
        x = self.up1(x3, t)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x, t)
        x = torch.cat([x, x0], dim=1)
        x = self.out(x)
        return x


class DiffusionModel:
    def __init__(self, img_channels=3, img_size=64, device='cuda', timesteps=1000):
        self.device = device
        self.img_size = img_size
        self.img_channels = img_channels
        self.timesteps = timesteps

        # Define beta schedule
        self.beta = self.linear_beta_schedule(timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Initialize model
        self.model = UNet(img_channels=img_channels).to(device)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def forward_diffusion(self, x0, t):
        """Add noise to images"""
        noise = torch.randn_like(x0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

    @torch.no_grad()
    def sample(self, n_samples=16):
        """Generate images from noise"""
        self.model.eval()
        x = torch.randn((n_samples, self.img_channels, self.img_size, self.img_size)).to(self.device)

        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((n_samples,), i, dtype=torch.long, device=self.device)

            # Predict noise
            predicted_noise = self.model(x, t)

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Reverse diffusion step
            x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        self.model.train()
        return x

    def train(self, dataloader, epochs=100, lr=1e-4):
        """Train the diffusion model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
            epoch_loss = 0

            for batch in pbar:
                # Assume batch is either (images,) or (images, labels)
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                batch_size = images.shape[0]

                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                # Forward diffusion
                x_noisy, noise = self.forward_diffusion(images, t)

                # Predict noise
                predicted_noise = self.model(x_noisy, t)

                # Calculate loss
                loss = mse_loss(noise, predicted_noise)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'img_size': self.img_size,
            'img_channels': self.img_channels,
            'timesteps': self.timesteps
        }, path)

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# Example usage with MNIST
if __name__ == "__main__":
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    # Prepare MNIST dataset
    transform = transforms.Compose([
        transforms.Resize(32),  # Resize to 32x32 for faster training
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Downloading MNIST dataset...")
    train_dataset = datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    # Initialize diffusion model for grayscale images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    diffusion = DiffusionModel(img_channels=1, img_size=32, device=device, timesteps=500)

    # Train the model
    print("Starting training...")
    diffusion.train(dataloader, epochs=50, lr=2e-4)

    # Generate samples
    print("Generating samples...")
    samples = diffusion.sample(n_samples=16)

    # Save model
    diffusion.save('diffusion_mnist.pth')

    # Visualize generated samples
    samples = (samples.clamp(-1, 1) + 1) / 2  # Denormalize to [0, 1]
    samples = samples.cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('generated_mnist.png')
    print("Generated samples saved to 'generated_flower102.png'")
    print("Training complete! Model saved to 'diffusion_mnist.pth'")