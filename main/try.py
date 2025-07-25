import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


# =============================================================================
# ENHANCED NOISE SCHEDULING FOR BETTER QUALITY
# =============================================================================

class EnhancedNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu', schedule_type='cosine'):
        self.num_timesteps = num_timesteps
        self.device = device

        # Use cosine schedule for better quality
        if schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02, s=0.008):
        """Cosine schedule for better quality"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)

    def add_noise(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


# =============================================================================
# COMPLETELY FIXED U-NET COMPONENTS
# =============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Use LayerNorm instead of BatchNorm for stability
        self.norm1 = nn.GroupNorm(32 if in_channels >= 32 else in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(32 if out_channels >= 32 else out_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        # First block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None, None]

        # Second block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = min(num_heads, max(1, channels // 32))
        self.head_dim = channels // self.num_heads

        self.norm = nn.GroupNorm(32 if channels >= 32 else channels, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        out = self.to_out(out)

        return x + out


# =============================================================================
# COMPLETELY REDESIGNED STABLE U-NET
# =============================================================================

class StableConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256, num_classes=102, img_size=64):
        super().__init__()

        # Simple, stable channel configuration
        self.channels = [64, 128, 256, 512]
        self.img_size = img_size

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Class embedding
        self.class_embedding = nn.Sequential(
            nn.Embedding(num_classes, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)

        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        ch_in = self.channels[0]

        for i, ch_out in enumerate(self.channels):
            self.down_blocks.append(nn.ModuleList([
                ResnetBlock(ch_in, ch_out, time_emb_dim),
                ResnetBlock(ch_out, ch_out, time_emb_dim),
                SelfAttention(ch_out) if ch_out >= 256 else nn.Identity(),
                nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1) if i < len(self.channels) - 1 else nn.Identity()
            ]))
            ch_in = ch_out

        # Middle blocks
        self.mid_block1 = ResnetBlock(self.channels[-1], self.channels[-1], time_emb_dim)
        self.mid_attn = SelfAttention(self.channels[-1])
        self.mid_block2 = ResnetBlock(self.channels[-1], self.channels[-1], time_emb_dim)

        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        ch_in = self.channels[-1]

        for i, ch_out in enumerate(reversed(self.channels)):
            is_last = i == len(self.channels) - 1

            if not is_last:
                self.up_blocks.append(nn.ModuleList([
                    nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                    ResnetBlock(ch_out + ch_out, ch_out, time_emb_dim),  # +ch_out for skip connection
                    ResnetBlock(ch_out, ch_out, time_emb_dim),
                    SelfAttention(ch_out) if ch_out >= 256 else nn.Identity()
                ]))
            else:
                self.up_blocks.append(nn.ModuleList([
                    nn.Identity(),
                    ResnetBlock(ch_in + ch_out, ch_out, time_emb_dim),  # +ch_out for skip connection
                    ResnetBlock(ch_out, ch_out, time_emb_dim),
                    nn.Identity()
                ]))

            ch_in = ch_out

        # Final output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32 if self.channels[0] >= 32 else self.channels[0], self.channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channels[0], out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t, class_labels):
        # Embeddings
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_embedding(class_labels)
        emb = t_emb + c_emb

        # Initial conv
        x = self.init_conv(x)

        # Encoder
        skip_connections = []
        for resnet1, resnet2, attn, downsample in self.down_blocks:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            skip_connections.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # Decoder
        for i, (upsample, resnet1, resnet2, attn) in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]

            x = upsample(x)

            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)

            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)

        return self.final_conv(x)


# =============================================================================
# FIXED DISCRIMINATOR WITH LOGITS OUTPUT
# =============================================================================

class StableDiscriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64, num_classes=102):
        super().__init__()

        # Simple discriminator without final sigmoid (for logits)
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 512)

        # Final classifier (outputs logits, not probabilities)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
            # No sigmoid here - we'll use BCEWithLogitsLoss
        )

    def forward(self, x, class_labels=None):
        features = self.main(x)

        if class_labels is not None:
            # Add class conditioning
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.view(class_emb.size(0), class_emb.size(1), 1, 1)
            class_emb = class_emb.expand(-1, -1, features.size(2), features.size(3))
            features = features + class_emb

        return self.classifier(features).squeeze()


# =============================================================================
# DATASET LOADER
# =============================================================================

class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
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


def create_dataloaders(dataset_path, batch_size=8, img_size=64, num_workers=2):
    """Create dataloaders"""

    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.12), int(img_size * 1.12))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
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
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
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
# STABLE DIFFUSION GENERATOR
# =============================================================================

class StableDiffusionGenerator(nn.Module):
    def __init__(self, img_channels=3, img_size=64, num_classes=102, device='cpu'):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.device = device

        self.noise_scheduler = EnhancedNoiseScheduler(device=device, schedule_type='cosine')
        self.unet = StableConditionalUNet(
            in_channels=img_channels,
            out_channels=img_channels,
            num_classes=num_classes,
            img_size=img_size
        )

    def ddim_sample(self, batch_size, device, class_labels=None, num_inference_steps=50, eta=0.0):
        """DDIM sampling"""
        x = torch.randn(batch_size, self.img_channels, self.img_size, self.img_size, device=device)

        if class_labels is None:
            class_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)

        timesteps = torch.linspace(self.noise_scheduler.num_timesteps - 1, 0, num_inference_steps, device=device).long()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                predicted_noise = self.unet(x, t_batch, class_labels)

                alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]

                if i < len(timesteps) - 1:
                    alpha_cumprod_t_prev = self.noise_scheduler.alphas_cumprod[timesteps[i + 1]]
                else:
                    alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

                # Predict x0
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                # DDIM step
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt

        return torch.clamp(x, -1, 1)

    def forward(self, batch_size, device, class_labels=None, num_inference_steps=50, use_ddim=True):
        """Generate images"""
        if use_ddim:
            return self.ddim_sample(batch_size, device, class_labels, num_inference_steps)
        else:
            return self.ddpm_sample(batch_size, device, class_labels, num_inference_steps)

    def ddpm_sample(self, batch_size, device, class_labels=None, num_inference_steps=50):
        """Original DDPM sampling"""
        x = torch.randn(batch_size, self.img_channels, self.img_size, self.img_size, device=device)

        if class_labels is None:
            class_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)

        timesteps = torch.linspace(self.noise_scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long,
                                   device=device)

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
# FIXED LOSS FUNCTIONS (AUTOCAST SAFE)
# =============================================================================

def discriminator_loss(real_logits, fake_logits):
    """Discriminator loss using BCEWithLogitsLoss (autocast safe)"""
    real_loss = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits) * 0.9
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits) + 0.1
    )
    return (real_loss + fake_loss) / 2


def generator_adversarial_loss(fake_logits):
    """Generator adversarial loss using BCEWithLogitsLoss (autocast safe)"""
    return F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )


def diffusion_loss(noise_scheduler, unet, x_0, t, class_labels):
    """Diffusion training loss"""
    noise = torch.randn_like(x_0)
    x_t = noise_scheduler.add_noise(x_0, t, noise)
    predicted_noise = unet(x_t, t, class_labels)
    return F.mse_loss(predicted_noise, noise)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_stable_diffusion_gan(dataset_path, num_epochs=60, batch_size=6, img_size=64,
                               device='cuda', lr=1e-4, save_interval=10):
    """Stable training loop"""

    train_loader, val_loader, num_classes = create_dataloaders(
        dataset_path, batch_size=batch_size, img_size=img_size, num_workers=2
    )
    print(f"Training on {num_classes} flower classes")

    # Initialize models
    generator = StableDiffusionGenerator(
        img_channels=3, img_size=img_size, num_classes=num_classes, device=device
    ).to(device)

    discriminator = StableDiscriminator(
        img_channels=3, img_size=img_size, num_classes=num_classes
    ).to(device)

    # Optimizers
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    # Schedulers
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=num_epochs)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=num_epochs)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    best_loss = float('inf')

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_diff_loss = 0

        for batch_idx, (real_images, class_labels) in enumerate(train_loader):
            current_batch_size = real_images.size(0)
            real_images = real_images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            # Train Discriminator
            d_optimizer.zero_grad()

            if device == 'cuda' and scaler:
                with torch.amp.autocast('cuda'):
                    fake_images = generator(current_batch_size, device, class_labels, num_inference_steps=20,
                                            use_ddim=True)
                    real_logits = discriminator(real_images, class_labels)
                    fake_logits = discriminator(fake_images.detach(), class_labels)
                    d_loss = discriminator_loss(real_logits, fake_logits)

                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                fake_images = generator(current_batch_size, device, class_labels, num_inference_steps=20)
                real_logits = discriminator(real_images, class_labels)
                fake_logits = discriminator(fake_images.detach(), class_labels)
                d_loss = discriminator_loss(real_logits, fake_logits)
                d_loss.backward()
                d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            if device == 'cuda' and scaler:
                with torch.amp.autocast('cuda'):
                    # Diffusion loss
                    t = torch.randint(0, generator.noise_scheduler.num_timesteps, (current_batch_size,), device=device)
                    diff_loss = diffusion_loss(
                        generator.noise_scheduler, generator.unet, real_images, t, class_labels
                    )

                    # Adversarial loss (occasionally)
                    if batch_idx % 5 == 0:
                        fake_images_for_g = generator(current_batch_size, device, class_labels, num_inference_steps=20,
                                                      use_ddim=True)
                        fake_logits_for_g = discriminator(fake_images_for_g, class_labels)
                        adv_loss = generator_adversarial_loss(fake_logits_for_g)
                        g_loss = diff_loss + 0.02 * adv_loss
                    else:
                        g_loss = diff_loss
                        adv_loss = torch.tensor(0.0, device=device)

                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                t = torch.randint(0, generator.noise_scheduler.num_timesteps, (current_batch_size,), device=device)
                diff_loss = diffusion_loss(
                    generator.noise_scheduler, generator.unet, real_images, t, class_labels
                )

                if batch_idx % 5 == 0:
                    fake_images_for_g = generator(current_batch_size, device, class_labels, num_inference_steps=20)
                    fake_logits_for_g = discriminator(fake_images_for_g, class_labels)
                    adv_loss = generator_adversarial_loss(fake_logits_for_g)
                    g_loss = diff_loss + 0.02 * adv_loss
                else:
                    g_loss = diff_loss
                    adv_loss = torch.tensor(0.0, device=device)

                g_loss.backward()
                g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_diff_loss += diff_loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                      f"Diff_loss: {diff_loss.item():.4f} Adv_loss: {adv_loss.item():.4f}")

            # Memory cleanup
            if device == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()

        # Epoch summary
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_diff_loss = epoch_diff_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg D_loss: {avg_d_loss:.4f}, "
              f"Avg G_loss: {avg_g_loss:.4f}, Avg Diff_loss: {avg_diff_loss:.4f}")
        print(f"LR - Generator: {g_scheduler.get_last_lr()[0]:.6f}, Discriminator: {d_scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch,
                'num_classes': num_classes,
                'best_loss': best_loss
            }, 'Trial Output-1/best_stable_diffusion_gan.pth')

        # Periodic saves and samples
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch,
                'num_classes': num_classes
            }, f'stable_diffusion_gan_epoch_{epoch + 1}.pth')

            generate_samples(generator, device, num_classes, epoch + 1)

    return generator, discriminator


# =============================================================================
# SAMPLE GENERATION AND DATASET EXPANSION
# =============================================================================

def generate_samples(generator, device, num_classes, epoch, num_samples=16):
    """Generate sample images"""
    generator.eval()

    with torch.no_grad():
        class_labels = torch.arange(min(num_samples, num_classes), device=device) % num_classes
        samples = generator(len(class_labels), device, class_labels, num_inference_steps=50, use_ddim=True)
        samples = (samples + 1) / 2  # Denormalize

        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_title(f'Class {class_labels[i].item() + 1}', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'stable_samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Sample images saved: stable_samples_epoch_{epoch}.png")


def expand_dataset_stable(generator, device, dataset_path, num_classes, samples_per_class=100):
    """Generate images for dataset expansion"""
    generator.eval()
    output_dir = os.path.join(dataset_path, 'generated_stable')
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for class_id in range(num_classes):
            class_dir = os.path.join(output_dir, str(class_id + 1))
            os.makedirs(class_dir, exist_ok=True)

            print(f"Generating {samples_per_class} images for class {class_id + 1}")

            batch_size = 4
            generated_count = 0

            while generated_count < samples_per_class:
                current_batch_size = min(batch_size, samples_per_class - generated_count)
                class_labels = torch.full((current_batch_size,), class_id, device=device)

                images = generator(current_batch_size, device, class_labels,
                                   num_inference_steps=75, use_ddim=True)
                images = (images + 1) / 2  # Denormalize

                for i, img in enumerate(images):
                    img_path = os.path.join(class_dir, f'stable_{generated_count + i:05d}.png')
                    img_pil = transforms.ToPILImage()(img.cpu())
                    img_pil.save(img_path, format='PNG')

                generated_count += current_batch_size

                # Memory cleanup
                if device == 'cuda':
                    torch.cuda.empty_cache()

    print(f"Stable dataset expansion complete! Generated images saved in {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Dataset path - modify this to your actual path
    DATASET_PATH = r"../dataset"

    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # Optimize for GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
        print("Using CPU - Training will be slower")

    # Training parameters
    if device == 'cuda':
        NUM_EPOCHS = 60
        BATCH_SIZE = 6
        IMG_SIZE = 64
        LEARNING_RATE = 1e-4
        SAVE_INTERVAL = 10
        SAMPLES_PER_CLASS = 100
    else:
        NUM_EPOCHS = 30
        BATCH_SIZE = 4
        IMG_SIZE = 32
        LEARNING_RATE = 1e-4
        SAVE_INTERVAL = 10
        SAMPLES_PER_CLASS = 50

    print(f"\n=== STABLE TRAINING CONFIGURATION ===")
    print(f"Device: {device}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Mixed Precision: {'Enabled' if device == 'cuda' else 'Disabled'}")
    print("=" * 40)

    try:
        print("\nStarting stable training...")
        generator, discriminator = train_stable_diffusion_gan(
            dataset_path=DATASET_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=device,
            lr=LEARNING_RATE,
            save_interval=SAVE_INTERVAL
        )

        print("\nExpanding dataset...")
        expand_dataset_stable(
            generator=generator,
            device=device,
            dataset_path=DATASET_PATH,
            num_classes=102,
            samples_per_class=SAMPLES_PER_CLASS
        )

        print("Training and dataset expansion completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()

        if device == 'cuda':
            torch.cuda.empty_cache()
