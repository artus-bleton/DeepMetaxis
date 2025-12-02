import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Encodeur ----------
class Encoder(nn.Module):
    """Encode (B,3,64,64) -> (B, latent_dim)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        c, h, w = input_dim  # (3,64,64)
        assert (c, h, w) == (3, 64, 64), "Cet encodeur suppose des images 3x64x64."

        self.conv = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        # x: (B,C,H,W)
        h = self.conv(x)              # (B, 128*8*8)
        z = self.fc(h)                # (B, latent_dim)
        return z


# ---------- Décodeur ----------
class Decoder(nn.Module):
    """Decode (B,latent_dim) -> (B,3,64,64)"""
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        c, h, w = output_shape  # (3,64,64)
        assert (c, h, w) == (3, 64, 64), "Ce décodeur reconstruit des images 3x64x64."

        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)

        self.deconv = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # suppose des pixels normalisés en [0,1]
        )

    def forward(self, z):
        x = F.relu(self.fc(z), inplace=True)
        x = x.view(z.size(0), 128, 8, 8)   # (B,128,8,8)
        x = self.deconv(x)                 # (B,3,64,64)
        return x


# ---------- Dynamique latente ----------
class LatentDynamics(nn.Module):
    """Prévoit z_{t+1} à partir de z_t et a_t (résiduel : z_{t+1} = z_t + Δz)"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z, a):
        """
        z: (B, latent_dim)
        a: (B, action_dim) one-hot
        """
        h = torch.cat([z, a], dim=-1)  # (B, latent_dim + action_dim)
        dz = self.net(h)               # (B, latent_dim)
        return z + dz                  # résiduel


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
        # z: (B, latent_dim) ou (1, latent_dim)
        return self.fc(z)
