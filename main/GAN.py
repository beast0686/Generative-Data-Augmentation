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
# ENHANCED U-NET COMPONENTS FOR MAXIMUM REALISM
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

        # Enhanced normalization for better quality
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


class EnhancedSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = min(num_heads, max(1, channels // 16))  # More heads for better quality
        self.head_dim = channels // self.num_heads

        self.norm = nn.GroupNorm(32 if channels >= 32 else channels, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Dropout(0.05)  # Light dropout for regularization
        )
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

        # Compute attention with improved stability
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.05, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        out = self.to_out(out)

        return x + out


# =============================================================================
# ULTRA-REALISTIC U-NET ARCHITECTURE
# =============================================================================

class UltraRealisticConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=512, num_classes=102, img_size=64):
        super().__init__()

        # Enhanced channel configuration for better quality
        self.channels = [64, 128, 256, 512, 768]
        self.img_size = img_size

        # Enhanced time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # Enhanced class embedding
        self.class_embedding = nn.Sequential(
            nn.Embedding(num_classes, time_emb_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution with better initialization
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], 7, padding=3)  # Larger kernel for better features

        # Enhanced encoder blocks
        self.down_blocks = nn.ModuleList()
        ch_in = self.channels[0]

        for i, ch_out in enumerate(self.channels):
            use_attention = ch_out >= 256

            self.down_blocks.append(nn.ModuleList([
                ResnetBlock(ch_in, ch_out, time_emb_dim, dropout=0.1),
                ResnetBlock(ch_out, ch_out, time_emb_dim, dropout=0.1),
                ResnetBlock(ch_out, ch_out, time_emb_dim, dropout=0.05),  # Extra block for quality
                EnhancedSelfAttention(ch_out) if use_attention else nn.Identity(),
                nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1) if i < len(self.channels) - 1 else nn.Identity()
            ]))
            ch_in = ch_out

        # Enhanced middle blocks
        mid_ch = self.channels[-1]
        self.mid_block1 = ResnetBlock(mid_ch, mid_ch, time_emb_dim, dropout=0.1)
        self.mid_attn1 = EnhancedSelfAttention(mid_ch)
        self.mid_block2 = ResnetBlock(mid_ch, mid_ch, time_emb_dim, dropout=0.1)
        self.mid_attn2 = EnhancedSelfAttention(mid_ch)
        self.mid_block3 = ResnetBlock(mid_ch, mid_ch, time_emb_dim, dropout=0.05)

        # Enhanced decoder blocks
        self.up_blocks = nn.ModuleList()
        ch_in = self.channels[-1]

        for i, ch_out in enumerate(reversed(self.channels)):
            is_last = i == len(self.channels) - 1
            use_attention = ch_out >= 256

            if not is_last:
                self.up_blocks.append(nn.ModuleList([
                    nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                    ResnetBlock(ch_out + ch_out, ch_out, time_emb_dim, dropout=0.1),
                    ResnetBlock(ch_out, ch_out, time_emb_dim, dropout=0.1),
                    ResnetBlock(ch_out, ch_out, time_emb_dim, dropout=0.05),  # Extra block
                    EnhancedSelfAttention(ch_out) if use_attention else nn.Identity()
                ]))
            else:
                self.up_blocks.append(nn.ModuleList([
                    nn.Identity(),
                    ResnetBlock(ch_in + ch_out, ch_out, time_emb_dim, dropout=0.1),
                    ResnetBlock(ch_out, ch_out, time_emb_dim, dropout=0.1),
                    ResnetBlock(ch_out, ch_out, time_emb_dim, dropout=0.05),
                    nn.Identity()
                ]))

            ch_in = ch_out

        # Enhanced final output layers
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32 if self.channels[0] >= 32 else self.channels[0], self.channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channels[0], self.channels[0] // 2, 3, padding=1),
            nn.GroupNorm(16 if self.channels[0] >= 32 else self.channels[0] // 2, self.channels[0] // 2),
            nn.SiLU(),
            nn.Conv2d(self.channels[0] // 2, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t, class_labels):
        # Enhanced embeddings
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_embedding(class_labels)
        emb = t_emb + c_emb

        # Initial conv
        x = self.init_conv(x)

        # Enhanced encoder
        skip_connections = []
        for resnet1, resnet2, resnet3, attn, downsample in self.down_blocks:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = resnet3(x, emb)
            x = attn(x)
            skip_connections.append(x)
            x = downsample(x)

        # Enhanced middle
        x = self.mid_block1(x, emb)
        x = self.mid_attn1(x)
        x = self.mid_block2(x, emb)
        x = self.mid_attn2(x)
        x = self.mid_block3(x, emb)

        # Enhanced decoder
        for i, (upsample, resnet1, resnet2, resnet3, attn) in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]

            x = upsample(x)

            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)

            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = resnet3(x, emb)
            x = attn(x)

        return self.final_conv(x)


# =============================================================================
# ENHANCED DISCRIMINATOR FOR BETTER TRAINING
# =============================================================================

class UltraRealisticDiscriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64, num_classes=102):
        super().__init__()

        # Progressive discriminator for better quality assessment
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            # 4x4 -> 2x2
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Enhanced class embedding
        self.class_embedding = nn.Sequential(
            nn.Embedding(num_classes, 1024),
            nn.LeakyReLU(0.2)
        )

        # Multi-scale classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x, class_labels=None):
        features = self.main(x)

        if class_labels is not None:
            # Enhanced class conditioning
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.view(class_emb.size(0), class_emb.size(1), 1, 1)
            class_emb = class_emb.expand(-1, -1, features.size(2), features.size(3))
            features = features + class_emb

        return self.classifier(features).squeeze()


# =============================================================================
# DATASET LOADER WITH ENHANCED PREPROCESSING
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


def create_enhanced_dataloaders(dataset_path, batch_size=4, img_size=64, num_workers=2):
    """Create enhanced dataloaders for maximum quality"""

    # Enhanced training transforms for realism
    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.25), int(img_size * 1.25))),  # Larger resize
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # Light vertical flip
        transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # High-quality validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
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
# ULTRA-REALISTIC DIFFUSION GENERATOR
# =============================================================================

class UltraRealisticDiffusionGenerator(nn.Module):
    def __init__(self, img_channels=3, img_size=64, num_classes=102, device='cpu'):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.device = device

        self.noise_scheduler = EnhancedNoiseScheduler(device=device, schedule_type='cosine')
        self.unet = UltraRealisticConditionalUNet(
            in_channels=img_channels,
            out_channels=img_channels,
            num_classes=num_classes,
            img_size=img_size
        )

    def ddim_sample(self, batch_size, device, class_labels=None, num_inference_steps=100, eta=0.0):
        """Enhanced DDIM sampling for maximum quality"""
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

                # Enhanced x0 prediction with clamping
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                # DDIM step with optional stochasticity
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta ** 2) * predicted_noise
                random_noise = torch.randn_like(x) * eta * torch.sqrt(1 - alpha_cumprod_t_prev)
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + random_noise

        return torch.clamp(x, -1, 1)

    def forward(self, batch_size, device, class_labels=None, num_inference_steps=100, use_ddim=True):
        """Generate ultra-realistic images"""
        if use_ddim:
            return self.ddim_sample(batch_size, device, class_labels, num_inference_steps)
        else:
            return self.ddpm_sample(batch_size, device, class_labels, num_inference_steps)

    def ddpm_sample(self, batch_size, device, class_labels=None, num_inference_steps=100):
        """Enhanced DDPM sampling"""
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
# ENHANCED LOSS FUNCTIONS
# =============================================================================

def enhanced_discriminator_loss(real_logits, fake_logits):
    """Enhanced discriminator loss with label smoothing and noise"""
    # Label smoothing for better training stability
    real_labels = torch.ones_like(real_logits) * 0.9 + torch.rand_like(real_logits) * 0.1
    fake_labels = torch.zeros_like(fake_logits) + torch.rand_like(fake_logits) * 0.1

    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
    return (real_loss + fake_loss) / 2


def enhanced_generator_adversarial_loss(fake_logits):
    """Enhanced generator adversarial loss"""
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def enhanced_diffusion_loss(noise_scheduler, unet, x_0, t, class_labels):
    """Enhanced diffusion training loss with perceptual components"""
    noise = torch.randn_like(x_0)
    x_t = noise_scheduler.add_noise(x_0, t, noise)
    predicted_noise = unet(x_t, t, class_labels)

    # Combination of MSE and L1 loss for better quality
    mse_loss = F.mse_loss(predicted_noise, noise)
    l1_loss = F.l1_loss(predicted_noise, noise)
    return 0.8 * mse_loss + 0.2 * l1_loss


# =============================================================================
# ULTRA-REALISTIC TRAINING LOOP
# =============================================================================

def train_ultra_realistic_diffusion_gan(dataset_path, num_epochs=100, batch_size=4, img_size=64,
                                        device='cuda', lr=5e-5, save_interval=5):
    """Ultra-realistic training loop with enhanced features"""

    train_loader, val_loader, num_classes = create_enhanced_dataloaders(
        dataset_path, batch_size=batch_size, img_size=img_size, num_workers=2
    )
    print(f"Training ultra-realistic model on {num_classes} flower classes")

    # Initialize enhanced models
    generator = UltraRealisticDiffusionGenerator(
        img_channels=3, img_size=img_size, num_classes=num_classes, device=device
    ).to(device)

    discriminator = UltraRealisticDiscriminator(
        img_channels=3, img_size=img_size, num_classes=num_classes
    ).to(device)

    # Enhanced optimizers with different learning rates
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr * 1.5, betas=(0.9, 0.999), weight_decay=0.01)

    # Enhanced schedulers
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(g_optimizer, T_0=20, T_mult=2)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(d_optimizer, T_0=20, T_mult=2)

    # Mixed precision for efficiency
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    best_loss = float('inf')
    patience = 0
    max_patience = 15

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
                    fake_images = generator(current_batch_size, device, class_labels,
                                            num_inference_steps=25, use_ddim=True)
                    real_logits = discriminator(real_images, class_labels)
                    fake_logits = discriminator(fake_images.detach(), class_labels)
                    d_loss = enhanced_discriminator_loss(real_logits, fake_logits)

                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                fake_images = generator(current_batch_size, device, class_labels, num_inference_steps=25)
                real_logits = discriminator(real_images, class_labels)
                fake_logits = discriminator(fake_images.detach(), class_labels)
                d_loss = enhanced_discriminator_loss(real_logits, fake_logits)
                d_loss.backward()
                d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            if device == 'cuda' and scaler:
                with torch.amp.autocast('cuda'):
                    # Enhanced diffusion loss
                    t = torch.randint(0, generator.noise_scheduler.num_timesteps, (current_batch_size,), device=device)
                    diff_loss = enhanced_diffusion_loss(
                        generator.noise_scheduler, generator.unet, real_images, t, class_labels
                    )

                    # Adversarial loss (less frequent for stability)
                    if batch_idx % 7 == 0:  # Even less frequent
                        fake_images_for_g = generator(current_batch_size, device, class_labels,
                                                      num_inference_steps=25, use_ddim=True)
                        fake_logits_for_g = discriminator(fake_images_for_g, class_labels)
                        adv_loss = enhanced_generator_adversarial_loss(fake_logits_for_g)
                        g_loss = diff_loss + 0.01 * adv_loss  # Lower weight
                    else:
                        g_loss = diff_loss
                        adv_loss = torch.tensor(0.0, device=device)

                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                t = torch.randint(0, generator.noise_scheduler.num_timesteps, (current_batch_size,), device=device)
                diff_loss = enhanced_diffusion_loss(
                    generator.noise_scheduler, generator.unet, real_images, t, class_labels
                )

                if batch_idx % 7 == 0:
                    fake_images_for_g = generator(current_batch_size, device, class_labels, num_inference_steps=25)
                    fake_logits_for_g = discriminator(fake_images_for_g, class_labels)
                    adv_loss = enhanced_generator_adversarial_loss(fake_logits_for_g)
                    g_loss = diff_loss + 0.01 * adv_loss
                else:
                    g_loss = diff_loss
                    adv_loss = torch.tensor(0.0, device=device)

                g_loss.backward()
                g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_diff_loss += diff_loss.item()

            if batch_idx % 25 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                      f"Diff_loss: {diff_loss.item():.4f} Adv_loss: {adv_loss.item():.4f}")

            # Memory management
            if device == 'cuda' and batch_idx % 50 == 0:
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
        print(f"Current LR - Generator: {g_scheduler.get_last_lr()[0]:.7f}, "
              f"Discriminator: {d_scheduler.get_last_lr()[0]:.7f}")

        # Enhanced model saving with early stopping
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            patience = 0
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
                'num_classes': num_classes,
                'best_loss': best_loss
            }, 'best_ultra_realistic_diffusion_gan.pth')
            print(f"✓ New best model saved with loss: {best_loss:.4f}")
        else:
            patience += 1

        # Periodic saves and samples
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch,
                'num_classes': num_classes
            }, f'ultra_realistic_epoch_{epoch + 1}.pth')

            generate_ultra_realistic_samples(generator, device, num_classes, epoch + 1)

        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

    return generator, discriminator


# =============================================================================
# ULTRA-REALISTIC SAMPLE GENERATION
# =============================================================================

def generate_ultra_realistic_samples(generator, device, num_classes, epoch, num_samples=16):
    """Generate ultra-realistic sample images with maximum quality settings"""
    generator.eval()

    with torch.no_grad():
        class_labels = torch.arange(min(num_samples, num_classes), device=device) % num_classes

        # Generate with maximum quality settings
        samples = generator(len(class_labels), device, class_labels,
                            num_inference_steps=150, use_ddim=True)
        samples = (samples + 1) / 2  # Denormalize

        # Create output directory
        output_dir = f'ultra_realistic_samples_epoch_{epoch}'
        os.makedirs(output_dir, exist_ok=True)

        # Save individual high-quality images
        for i, img in enumerate(samples):
            img_path = os.path.join(output_dir, f'ultra_realistic_class_{class_labels[i].item() + 1:03d}_{i:03d}.png')
            # Convert to PIL and enhance
            img_pil = transforms.ToPILImage()(img.cpu())
            # Upscale for better quality
            img_pil = img_pil.resize((128, 128), Image.LANCZOS)
            img_pil.save(img_path, format='PNG', optimize=False, compress_level=0)

        # Create grid visualization
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_title(f'Class {class_labels[i].item() + 1}', fontsize=12, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/grid_ultra_realistic_epoch_{epoch}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Ultra-realistic samples saved in: {output_dir}")


def generate_production_quality_images(generator, device, dataset_path, num_classes,
                                       samples_per_class=50, target_classes=None):
    """Generate production-quality images for specific classes"""
    generator.eval()

    # Create high-quality output directory
    output_dir = os.path.join(dataset_path, 'production_quality')
    os.makedirs(output_dir, exist_ok=True)

    # If specific classes aren't provided, generate for all classes
    if target_classes is None:
        target_classes = range(num_classes)

    print(f"🎨 Generating production-quality images for {len(target_classes)} classes...")

    with torch.no_grad():
        for class_id in target_classes:
            class_dir = os.path.join(output_dir, f'class_{class_id + 1:03d}')
            os.makedirs(class_dir, exist_ok=True)

            print(f"  📸 Class {class_id + 1}: Generating {samples_per_class} images...")

            for i in range(samples_per_class):
                class_labels = torch.tensor([class_id], device=device)

                # Generate single image with maximum quality settings
                image = generator(1, device, class_labels,
                                  num_inference_steps=200, use_ddim=True)  # Maximum steps
                image = (image + 1) / 2  # Denormalize

                # Save with highest quality
                img_path = os.path.join(class_dir, f'prod_quality_{i:04d}.png')
                img_pil = transforms.ToPILImage()(image[0].cpu())

                # Enhanced post-processing for maximum realism
                img_pil = img_pil.resize((256, 256), Image.LANCZOS)  # High-res upscale
                img_pil.save(img_path, format='PNG', optimize=False, compress_level=0, dpi=(300, 300))

                if (i + 1) % 10 == 0:
                    print(f"    ✓ Generated {i + 1}/{samples_per_class} images")

                # Clear memory after each image
                if device == 'cuda':
                    torch.cuda.empty_cache()

    print(f"🎉 Production-quality images complete! Saved in: {output_dir}")


def expand_dataset_ultra_realistic(generator, device, dataset_path, num_classes, samples_per_class=200):
    """Generate ultra-realistic images for comprehensive dataset expansion"""
    generator.eval()
    output_dir = os.path.join(dataset_path, 'ultra_realistic_expansion')
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Starting ultra-realistic dataset expansion...")
    print(
        f"📊 Target: {samples_per_class} images per class × {num_classes} classes = {samples_per_class * num_classes} total images")

    with torch.no_grad():
        for class_id in range(num_classes):
            class_dir = os.path.join(output_dir, str(class_id + 1))
            os.makedirs(class_dir, exist_ok=True)

            print(f"🌸 Class {class_id + 1}/{num_classes}: Generating {samples_per_class} ultra-realistic images...")

            batch_size = 2  # Small batch for maximum quality
            generated_count = 0

            while generated_count < samples_per_class:
                current_batch_size = min(batch_size, samples_per_class - generated_count)
                class_labels = torch.full((current_batch_size,), class_id, device=device)

                # Generate with ultra-high quality settings
                images = generator(current_batch_size, device, class_labels,
                                   num_inference_steps=175, use_ddim=True)
                images = (images + 1) / 2

                for i, img in enumerate(images):
                    img_path = os.path.join(class_dir, f'ultra_{generated_count + i:05d}.png')
                    img_pil = transforms.ToPILImage()(img.cpu())
                    # High-quality upscaling
                    img_pil = img_pil.resize((128, 128), Image.LANCZOS)
                    img_pil.save(img_path, format='PNG', optimize=False, dpi=(300, 300))

                generated_count += current_batch_size

                if generated_count % 20 == 0:
                    print(
                        f"  ✓ Progress: {generated_count}/{samples_per_class} ({generated_count / samples_per_class * 100:.1f}%)")

                # Aggressive memory management
                if device == 'cuda':
                    torch.cuda.empty_cache()

    print(f"🎊 Ultra-realistic dataset expansion complete!")
    print(f"📁 Generated images saved in: {output_dir}")


# =============================================================================
# MAIN EXECUTION WITH MAXIMUM REALISM SETTINGS
# =============================================================================

if __name__ == "__main__":
    # Dataset path - modify this to your actual path
    DATASET_PATH = r"../dataset"

    # Device setup with optimization
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # Maximum GPU optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

        # Set high precision for quality
        torch.backends.cudnn.deterministic = False  # For speed
        torch.backends.cudnn.enabled = True

    else:
        device = 'cpu'
        print("⚠️ Using CPU - Training will be significantly slower")

    # Ultra-realistic training parameters
    if device == 'cuda':
        NUM_EPOCHS = 120  # Extended training for maximum quality
        BATCH_SIZE = 3  # Smaller for stability and quality
        IMG_SIZE = 64  # Optimal size for RTX 4060
        LEARNING_RATE = 3e-5  # Conservative for stable convergence
        SAVE_INTERVAL = 5  # Frequent saves
        SAMPLES_PER_CLASS = 250  # Large dataset expansion
    else:
        NUM_EPOCHS = 60
        BATCH_SIZE = 2
        IMG_SIZE = 32
        LEARNING_RATE = 1e-4
        SAVE_INTERVAL = 10
        SAMPLES_PER_CLASS = 100

    print(f"\n{'=' * 60}")
    print(f"🎨 ULTRA-REALISTIC FLOWER GENERATION SYSTEM")
    print(f"{'=' * 60}")
    print(f"🖥️  Device: {device.upper()}")
    print(f"📊 Epochs: {NUM_EPOCHS}")
    print(f"🔢 Batch Size: {BATCH_SIZE}")
    print(f"📐 Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"📈 Learning Rate: {LEARNING_RATE}")
    print(f"🎯 Target: Ultra-realistic flower images")
    print(f"🌸 Samples per Class: {SAMPLES_PER_CLASS}")
    print(f"🧠 Mixed Precision: {'✓ Enabled' if device == 'cuda' else '✗ Disabled'}")
    print(f"{'=' * 60}")

    try:
        print(f"\n🚀 Starting ultra-realistic training...")
        generator, discriminator = train_ultra_realistic_diffusion_gan(
            dataset_path=DATASET_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=device,
            lr=LEARNING_RATE,
            save_interval=SAVE_INTERVAL
        )

        print(f"\n🎨 Generating production-quality showcase images...")
        generate_production_quality_images(
            generator=generator,
            device=device,
            dataset_path=DATASET_PATH,
            num_classes=102,
            samples_per_class=25  # Showcase samples
        )

        print(f"\n📈 Expanding dataset with ultra-realistic images...")
        expand_dataset_ultra_realistic(
            generator=generator,
            device=device,
            dataset_path=DATASET_PATH,
            num_classes=102,
            samples_per_class=SAMPLES_PER_CLASS
        )

        print(f"\n🎉 ULTRA-REALISTIC GENERATION COMPLETE!")
        print(f"✓ Training completed successfully")
        print(f"✓ Production-quality images generated")
        print(f"✓ Dataset expanded with {SAMPLES_PER_CLASS * 102:,} new images")
        print(f"📁 Check output directories for results")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()

        if device == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared")
