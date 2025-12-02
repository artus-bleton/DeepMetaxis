import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Encodeur ----------
class Encoder(nn.Module):
    """Encode (B,3,64,64) -> (B,latent_dim)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        c, h, w = input_dim  # (3,64,64)
        assert (c, h, w) == (3, 64, 64), "Cet encodeur suppose des images 3x64x64."
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2),   # -> (32,31,31)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64,14,14)
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 14 * 14, latent_dim)

    def forward(self, x):
        # x: (B,C,H,W)
        return self.fc(self.conv(x))


# ---------- Décodeur ----------
class Decoder(nn.Module):
    """Decode (B,latent_dim) -> (B,3,64,64)"""
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        c, h, w = output_shape  # (3,64,64)
        assert (c, h, w) == (3, 64, 64), "Ce décodeur reconstruit des images 3x64x64."
        self.fc = nn.Linear(latent_dim, 64 * 14 * 14)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=1),  # 14→31
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2),                     # 31→64
            nn.Sigmoid()
        )

    def forward(self, z):
        x = F.relu(self.fc(z), inplace=True)
        x = x.view(z.size(0), 64, 14, 14)   # conserve la taille de batch
        return self.deconv(x)


# ---------- Dynamique latente ----------
class LatentDynamics(nn.Module):
    """Prévoit z_{t+1} à partir de z_t et a_t"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z, a):
        return self.fc(torch.cat([z, a], dim=-1))


# ---------- Politique ----------
class Policy(nn.Module):
    """Politique stochastique : z -> logits d'action"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, z):
        return self.fc(z)
