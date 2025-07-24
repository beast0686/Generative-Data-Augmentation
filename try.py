import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt

# =============================================================================
# NOISE SCHEDULING FOR DIFFUSION
# =============================================================================

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

# =============================================================================
# U-NET ARCHITECTURE COMPONENTS
# =============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)
        attn = torch.bmm(q, k.transpose(1, 2)) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.proj(out)
        return x + out

# =============================================================================
# CONDITIONAL U-NET WITH CLASS EMBEDDING
# =============================================================================

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128, num_classes=102, img_size=64):
        super().__init__()
        # Adjust channels based on image size
        if img_size <= 32:
            channels = [32, 64, 128]
        else:
            channels = [64, 128, 256, 512]
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, time_emb_dim)
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            downsample = nn.Conv2d(ch, ch, 3, stride=2, padding=1) if img_size > 16 or i < len(channels) - 1 else nn.Identity()
            self.encoder_blocks.append(nn.ModuleList([
                ResnetBlock(prev_ch, ch, time_emb_dim),
                ResnetBlock(ch, ch, time_emb_dim),
                AttentionBlock(ch) if ch >= 128 else nn.Identity(),
                downsample
            ]))
            prev_ch = ch
        # Middle
        self.middle_block1 = ResnetBlock(channels[-1], channels[-1], time_emb_dim)
        self.middle_attn = AttentionBlock(channels[-1]) if channels[-1] >= 128 else nn.Identity()
        self.middle_block2 = ResnetBlock(channels[-1], channels[-1], time_emb_dim)
        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            upsample = nn.ConvTranspose2d(prev_ch, ch, 4, stride=2, padding=1) if img_size > 16 or i < len(channels) - 1 else nn.Conv2d(prev_ch, ch, 3, padding=1)
            self.decoder_blocks.append(nn.ModuleList([
                upsample,
                ResnetBlock(ch * 2, ch, time_emb_dim),
                ResnetBlock(ch, ch, time_emb_dim),
                AttentionBlock(ch) if ch >= 128 else nn.Identity()
            ]))
            prev_ch = ch
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1)
        )

    def forward(self, x, t, class_labels):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_embedding(class_labels)
        emb = t_emb + c_emb
        x = self.init_conv(x)
        encoder_outputs = []
        for resnet1, resnet2, attn, downsample in self.encoder_blocks:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            encoder_outputs.append(x)
            x = downsample(x)
        x = self.middle_block1(x, emb)
        x = self.middle_attn(x)
        x = self.middle_block2(x, emb)
        for (upsample, resnet1, resnet2, attn), skip in zip(self.decoder_blocks, reversed(encoder_outputs)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
        return self.final_conv(x)

# =============================================================================
# ADAPTIVE DISCRIMINATOR
# =============================================================================

class AdaptiveDiscriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64):
        super().__init__()
        if img_size <= 32:
            features = [32, 64, 128]
            kernel_sizes = [3, 3, 3]
            strides = [2, 2, 1]
        else:
            features = [64, 128, 256, 512]
            kernel_sizes = [4, 4, 4, 4]
            strides = [2, 2, 2, 2]
        layers = []
        in_channels = img_channels
        for feature, kernel_size, stride in zip(features, kernel_sizes, strides):
            layers.append(
                nn.Conv2d(in_channels, feature, kernel_size, stride=stride, padding=1, bias=False)
            )
            if in_channels != img_channels:  # No batch norm for first layer
                layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = feature
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[-1], 1),
            nn.Sigmoid()
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

# =============================================================================
# DATASET LOADER FOR FLOWER DATASET
# =============================================================================

class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Custom dataset loader for flower dataset structure
        Args:
            root_dir: Path to dataset folder
            transform: Image transformations
            split: 'train', 'valid', or 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(self.root_dir)
                              if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            self.class_to_idx[class_name]
                        ))
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

def create_dataloaders(dataset_path, batch_size=32, img_size=64, num_workers=0):
    """Create train and validation dataloaders"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = FlowerDataset(dataset_path, transform=train_transform, split='train')
    val_dataset = FlowerDataset(dataset_path, transform=val_transform, split='valid')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader, len(train_dataset.classes)

# =============================================================================
# CONDITIONAL DIFFUSION GENERATOR
# =============================================================================

class ConditionalDiffusionGenerator(nn.Module):
    def __init__(self, img_channels=3, img_size=64, num_classes=102):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.noise_scheduler = NoiseScheduler()
        self.unet = ConditionalUNet(
            in_channels=img_channels,
            out_channels=img_channels,
            num_classes=num_classes,
            img_size=img_size
        )

    def forward(self, batch_size, device, class_labels=None, num_inference_steps=50):
        """Generate images using DDPM sampling with optional class conditioning"""
        x = torch.randn(batch_size, self.img_channels, self.img_size, self.img_size, device=device)
        if class_labels is None:
            class_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
        timesteps = torch.linspace(self.noise_scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                predicted_noise = self.unet(x, t_batch, class_labels)
                alpha = self.noise_scheduler.alphas[t]
                alpha_cumprod = self.noise_scheduler.alphas_cumprod[t]
                beta = self.noise_scheduler.betas[t]
                noise = torch.randn_like(x) if i < len(timesteps) - 1 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
                x = x + torch.sqrt(beta) * noise
        return torch.clamp(x, -1, 1)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def discriminator_loss(real_pred, fake_pred):
    """Binary cross-entropy loss for discriminator"""
    real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
    fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
    return (real_loss + fake_loss) / 2

def generator_adversarial_loss(fake_pred):
    """Adversarial loss for generator"""
    return F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

def conditional_diffusion_loss(noise_scheduler, unet, x_0, t, class_labels):
    """Conditional diffusion training loss"""
    noise = torch.randn_like(x_0)
    x_t = noise_scheduler.add_noise(x_0, t, noise)
    predicted_noise = unet(x_t, t, class_labels)
    return F.mse_loss(predicted_noise, noise)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def update_discriminator(optimizer, loss):
    """Update discriminator parameters"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_generator(optimizer, loss):
    """Update generator parameters"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_conditional_diffusion_gan(dataset_path, num_epochs=100, batch_size=32, img_size=64,
                                    device='cuda', lr=2e-4, save_interval=10):
    """Train the conditional diffusion GAN on the flower dataset"""
    train_loader, val_loader, num_classes = create_dataloaders(
        dataset_path, batch_size=batch_size, img_size=img_size
    )
    print(f"Training on {num_classes} flower classes")
    generator = ConditionalDiffusionGenerator(
        img_channels=3, img_size=img_size, num_classes=num_classes
    ).to(device)
    discriminator = AdaptiveDiscriminator(img_channels=3, img_size=img_size).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_diff_loss = 0
        for batch_idx, (real_images, class_labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            class_labels = class_labels.to(device)
            # Train Discriminator
            fake_images = generator(batch_size, device, class_labels, num_inference_steps=20)
            real_pred = discriminator(real_images)
            fake_pred = discriminator(fake_images.detach())
            d_loss = discriminator_loss(real_pred, fake_pred)
            update_discriminator(d_optimizer, d_loss)
            # Train Generator
            t = torch.randint(0, generator.noise_scheduler.num_timesteps, (batch_size,), device=device)
            diff_loss = conditional_diffusion_loss(
                generator.noise_scheduler, generator.unet, real_images, t, class_labels
            )
            fake_images_for_g = generator(batch_size, device, class_labels, num_inference_steps=20)
            fake_pred_for_g = discriminator(fake_images_for_g)
            adv_loss = generator_adversarial_loss(fake_pred_for_g)
            g_loss = diff_loss + 0.1 * adv_loss
            update_generator(g_optimizer, g_loss)
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_diff_loss += diff_loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                      f"Diff_loss: {diff_loss.item():.4f} Adv_loss: {adv_loss.item():.4f}")
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_diff_loss = epoch_diff_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg D_loss: {avg_d_loss:.4f}, "
              f"Avg G_loss: {avg_g_loss:.4f}, Avg Diff_loss: {avg_diff_loss:.4f}")
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
                'num_classes': num_classes
            }, f'diffusion_gan_epoch_{epoch + 1}.pth')
            generate_samples(generator, device, num_classes, epoch + 1)
    return generator, discriminator

# =============================================================================
# SAMPLE GENERATION AND VISUALIZATION
# =============================================================================

def generate_samples(generator, device, num_classes, epoch, num_samples=16):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        class_labels = torch.arange(min(num_samples, num_classes), device=device) % num_classes
        samples = generator(len(class_labels), device, class_labels, num_inference_steps=50)
        samples = (samples + 1) / 2
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(f'Class {class_labels[i].item() + 1}')
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'generated_samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

def expand_dataset(generator, device, dataset_path, num_classes, samples_per_class=50):
    """Generate new images to expand the dataset"""
    generator.eval()
    output_dir = os.path.join(dataset_path, 'generated')
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for class_id in range(num_classes):
            class_dir = os.path.join(output_dir, str(class_id + 1))
            os.makedirs(class_dir, exist_ok=True)
            print(f"Generating {samples_per_class} images for class {class_id + 1}")
            batch_size = 8
            generated_count = 0
            while generated_count < samples_per_class:
                current_batch_size = min(batch_size, samples_per_class - generated_count)
                class_labels = torch.full((current_batch_size,), class_id, device=device)
                images = generator(current_batch_size, device, class_labels, num_inference_steps=50)
                images = (images + 1) / 2
                for i, img in enumerate(images):
                    img_path = os.path.join(class_dir, f'generated_{generated_count + i:04d}.png')
                    transforms.ToPILImage()(img.cpu()).save(img_path)
                generated_count += current_batch_size
    print(f"Dataset expansion complete! Generated images saved in {output_dir}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    DATASET_PATH = r"D:\BNMIT\Internship\IAMPro2025\Generative-Data-Augmentation\dataset"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    NUM_EPOCHS = 20 if device == 'cpu' else 50
    BATCH_SIZE = 4 if device == 'cpu' else 16
    IMG_SIZE = 32 if device == 'cpu' else 64
    LEARNING_RATE = 2e-4
    try:
        print("Starting training...")
        generator, discriminator = train_conditional_diffusion_gan(
            dataset_path=DATASET_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=device,
            lr=LEARNING_RATE,
            save_interval=5
        )
        print("Expanding dataset...")
        expand_dataset(
            generator=generator,
            device=device,
            dataset_path=DATASET_PATH,
            num_classes=102,
            samples_per_class=20 if device == 'cpu' else 50
        )
        print("Training and dataset expansion completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()