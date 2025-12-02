import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import Encoder, Decoder, LatentDynamics, Policy


class DeepMetaxisAgent:
    """
    Monde latent (encoder/decoder + dynamics) + curiosité (erreur de dynamique)
    Politique séparée. Optimisations séparées: reconstruction, dynamique, policy.
    """
    def __init__(self, input_shape, latent_dim, action_dim,
                 beta=0.1, lambda_recon=1.0, ent_coef=0.01, device='cuda',
                 lr_world=1e-4, lr_policy=1e-4, weight_decay=0.0):

        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Pondérations / hyperparams
        self.beta = beta                   # poids curiosité
        self.lambda_recon = lambda_recon   # poids reconstruction
        self.ent_coef = ent_coef           # poids entropie

        # Modules
        self.encoder  = Encoder(input_shape, latent_dim).to(device)
        self.decoder  = Decoder(latent_dim, input_shape).to(device)
        self.dynamics = LatentDynamics(latent_dim, action_dim).to(device)
        self.policy   = Policy(latent_dim, action_dim).to(device)

        # Optimiseurs séparés
        self.optim_recon = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr_world, weight_decay=weight_decay
        )
        self.optim_dyn = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()),
            lr=lr_world, weight_decay=weight_decay
        )
        self.optim_policy = torch.optim.Adam(
            self.policy.parameters(), lr=lr_policy, weight_decay=weight_decay
        )

        # Stats online pour normaliser la curiosité (Welford)
        self._rint_mean = 0.0
        self._rint_m2   = 1e-6
        self._rint_cnt  = 1e-6

    # -------------------- utilitaires --------------------
    def _ensure_batch_obs(self, x):
        t = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return t.unsqueeze(0) if t.dim() == 3 else t

    def _one_hot(self, a_idx):
        # a_idx: (B,) long
        return F.one_hot(a_idx, num_classes=self.action_dim).float()

    def _update_rint_stats(self, x_scalar: float):
        self._rint_cnt += 1.0
        delta = x_scalar - self._rint_mean
        self._rint_mean += delta / self._rint_cnt
        delta2 = x_scalar - self._rint_mean
        self._rint_m2 += delta * delta2

    def _rint_norm(self, x_tensor):
        var = self._rint_m2 / max(self._rint_cnt - 1.0, 1.0)
        std = max(np.sqrt(var), 1e-6)
        x = (x_tensor - self._rint_mean) / std
        return torch.clamp(x, -3.0, 3.0)

    # -------------------- interaction --------------------
    def act(self, obs):
        obs = self._ensure_batch_obs(obs)
        with torch.no_grad():
            z = self.encoder(obs)
            logits = self.policy(z)
            a = torch.distributions.Categorical(logits=logits).sample()
        return a.item()

    # -------------------- monde latent --------------------
    def predict_latent(self, z, a_idx):
        """
        z: (B, latent_dim)
        a_idx: (B,) long ou scalaire long
        """
        if not torch.is_tensor(a_idx):
            a_idx = torch.tensor([a_idx], device=self.device, dtype=torch.long)
        if a_idx.dim() == 0:
            a_idx = a_idx.unsqueeze(0)
        a_oh = self._one_hot(a_idx).to(self.device)
        return self.dynamics(z, a_oh)

    def compute_intrinsic_reward(self, z, a_idx, z_next):
        """
        Erreur de prédiction latente moyenne par dimension (non normalisée).
        Retourne: (B,)
        """
        z_pred = self.predict_latent(z, a_idx)
        err = F.mse_loss(z_pred, z_next, reduction='none').mean(dim=-1)
        return err

    # -------------------- consolidation (sleep) --------------------
    def update_consolidation(self, batch):
        """
        batch = {'obs': (B,C,H,W), 'next_obs': (B,C,H,W), 'actions': (B,)}
        Étapes séparées :
          1) Reconstruction (encoder+decoder)   [loss = lambda_recon * MSE(x_hat, x)]
          2) Dynamique latente (encoder+dynamics) avec cible z_next détachée
        """
        obs      = torch.as_tensor(batch['obs'],      device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device, dtype=torch.float32)
        actions  = torch.as_tensor(batch['actions'],  device=self.device, dtype=torch.long)

        # ----- 1) Reconstruction -----
        z_recon  = self.encoder(obs)     # forward dédié
        x_hat    = self.decoder(z_recon)
        recon_loss = F.mse_loss(x_hat, obs)

        self.optim_recon.zero_grad()
        (self.lambda_recon * recon_loss).backward()
        self.optim_recon.step()

        # ----- 2) Dynamique latente -----
        z_dyn      = self.encoder(obs)         # forward indépendant
        z_next     = self.encoder(next_obs)
        z_next_det = z_next.detach()

        a_onehot = self._one_hot(actions)
        z_pred   = self.dynamics(z_dyn, a_onehot)
        dyn_loss = F.mse_loss(z_pred, z_next_det)

        self.optim_dyn.zero_grad()
        dyn_loss.backward()
        self.optim_dyn.step()

        return {
            'dyn_loss': float(dyn_loss.detach().cpu()),
            'recon_loss': float(recon_loss.detach().cpu())
        }

    # -------------------- phase wake (policy) --------------------
    def train_veil_step(self, obs, action, reward_ext, next_obs):
        """
        Mise à jour policy:
          - latent détaché (pas de fuite de gradient dans l'encodeur)
          - curiosité = MSE(z_pred, z_next) moyenne par dim, normalisée (Welford)
          - bonus d'entropie
          - avantage simple: R_ext + beta * R_int_norm
        """
        obs      = self._ensure_batch_obs(obs)
        next_obs = self._ensure_batch_obs(next_obs)
        action   = torch.as_tensor([action], device=self.device, dtype=torch.long)

        with torch.no_grad():
            z      = self.encoder(obs)
            z_next = self.encoder(next_obs)

        # Prédiction latente
        z_pred = self.predict_latent(z, action)

        # Curiosité (moyenne par dim) + MAJ stats + normalisation
        r_int = F.mse_loss(z_pred, z_next, reduction='none').mean(dim=-1)  # (1,)
        self._update_rint_stats(r_int.item())
        r_int_n = self._rint_norm(r_int).detach()

        # Politique (sur z détaché)
        logits  = self.policy(z.detach())
        logp    = F.log_softmax(logits, dim=-1).gather(1, action.view(-1, 1)).squeeze(1)  # (1,)
        entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(dim=-1)         # (1,)

        adv = torch.as_tensor([reward_ext], device=self.device, dtype=torch.float32) + self.beta * r_int_n
        loss = -(logp * adv).mean() - self.ent_coef * entropy.mean()

        self.optim_policy.zero_grad()
        loss.backward()
        self.optim_policy.step()

        return {
            'curiosity_raw': float(r_int.detach().cpu()),
            'curiosity_norm': float(r_int_n.detach().cpu()),
            'policy_loss': float(loss.detach().cpu()),
            'entropy': float(entropy.detach().cpu())
        }
