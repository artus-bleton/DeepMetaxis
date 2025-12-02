import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import Encoder, Decoder, LatentDynamics, Policy


class DeepMetaxisAgent:
    """
    Monde latent VAE (encoder/decoder + dynamics) + curiosité (erreur de dynamique)
    Politique séparée. Optimisations séparées:
      - VAE (reconstruction + KL + continuité temporelle)
      - Dynamique (prédiction de µ_{t+1} à partir de µ_t et a_t)
      - Policy
    """
    def __init__(self, input_shape, latent_dim, action_dim,
                 beta=0.1,
                 lambda_recon=1.0,
                 lambda_kl=1e-3,
                 lambda_cont=0.0,
                 ent_coef=0.01,
                 device='cuda',
                 lr_world=1e-4,
                 lr_policy=1e-4,
                 weight_decay=0.0):

        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Pondérations / hyperparams
        self.beta = beta                   # poids curiosité
        self.lambda_recon = lambda_recon   # poids reconstruction
        self.lambda_kl = lambda_kl         # poids KL VAE
        self.lambda_cont = lambda_cont     # poids continuité temporelle µ_t ~ µ_{t+1}
        self.ent_coef = ent_coef           # poids entropie

        # --- Modules ---
        # Encoder "backbone" (features) -> on ajoute µ/logvar par-dessus
        self.encoder = Encoder(input_shape, latent_dim).to(device)
        self.fc_mu = nn.Linear(latent_dim, latent_dim).to(device)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim).to(device)

        self.decoder  = Decoder(latent_dim, input_shape).to(device)
        self.dynamics = LatentDynamics(latent_dim, action_dim).to(device)
        self.policy   = Policy(latent_dim, action_dim).to(device)

        # --- Optimiseurs séparés ---
        # VAE: encoder + têtes µ/logvar + decoder
        self.optim_vae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.fc_mu.parameters()) +
            list(self.fc_logvar.parameters()) +
            list(self.decoder.parameters()),
            lr=lr_world,
            weight_decay=weight_decay
        )
        # Dynamique uniquement (encoder figé côté dyn)
        self.optim_dyn = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=lr_world,
            weight_decay=weight_decay
        )
        # Policy
        self.optim_policy = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr_policy,
            weight_decay=weight_decay
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

    # -------------------- VAE helpers --------------------
    def _encode(self, x, sample: bool = True):
        """
        x: (B,C,H,W)
        retourne:
          z: (B,d) échantillon ou µ (si sample=False)
          mu, logvar: (B,d)
        """
        h = self.encoder(x)            # (B,d')
        mu = self.fc_mu(h)             # (B,d)
        logvar = self.fc_logvar(h)     # (B,d)
        if sample:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar

    @staticmethod
    def _kl_normal(mu, logvar):
        # KL(q(z|x) || N(0,I))
        # moyenne sur batch
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # -------------------- interaction --------------------
    def act(self, obs):
        obs = self._ensure_batch_obs(obs)
        with torch.no_grad():
            # on utilise µ comme représentation déterministe pour la policy
            _, mu, _ = self._encode(obs, sample=False)
            logits = self.policy(mu)
            a = torch.distributions.Categorical(logits=logits).sample()
        return a.item()

    # -------------------- monde latent --------------------
    def predict_latent(self, z, a_idx):
        """
        z: (B, latent_dim)   (on prendra µ en pratique)
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
        z, z_next: en pratique µ_t, µ_{t+1}
        Retourne: (B,)
        """
        z_pred = self.predict_latent(z, a_idx)
        err = F.mse_loss(z_pred, z_next, reduction='none').mean(dim=-1)
        return err

    # -------------------- consolidation (sleep) --------------------
    def update_consolidation(self, batch):
        """
        batch = {'obs': (B,C,H,W), 'next_obs': (B,C,H,W), 'actions': (B,)}
        Deux phases:
          1) VAE: encoder+decoder avec recon (t, t+1) + KL (+ continuité temporelle sur µ)
          2) Dynamique: prédire µ_{t+1} à partir de µ_t et a_t (encoder figé)
        """
        obs      = torch.as_tensor(batch['obs'],      device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device, dtype=torch.float32)
        actions  = torch.as_tensor(batch['actions'],  device=self.device, dtype=torch.long)

        # ---------- 1) VAE (E, µ/logvar, D) ----------
        # encode présent & futur avec échantillonnage
        z_t, mu_t, logvar_t       = self._encode(obs,      sample=True)
        z_next, mu_next, logvar_next = self._encode(next_obs, sample=True)

        # recon présentes/futures
        x_hat_t    = self.decoder(z_t)
        x_hat_next = self.decoder(z_next)

        recon_t    = F.mse_loss(x_hat_t, obs)
        recon_next = F.mse_loss(x_hat_next, next_obs)
        recon_loss = recon_t + recon_next

        # KL pour t et t+1
        kl_t    = self._kl_normal(mu_t, logvar_t)
        kl_next = self._kl_normal(mu_next, logvar_next)
        kl_loss = 0.5 * (kl_t + kl_next)

        # continuité temporelle (sur µ)
        cont_loss = F.mse_loss(mu_next, mu_t) if self.lambda_cont > 0.0 else torch.tensor(0.0, device=self.device)

        vae_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_kl    * kl_loss +
            self.lambda_cont  * cont_loss
        )

        self.optim_vae.zero_grad()
        vae_loss.backward()
        self.optim_vae.step()

        # ---------- 2) Dynamique (f) ----------
        # On régénère des latents déterministes (µ) SANS gradient sur l'encodeur
        with torch.no_grad():
            _, mu_t_sg, _    = self._encode(obs,      sample=False)
            _, mu_next_sg, _ = self._encode(next_obs, sample=False)

        a_oh = self._one_hot(actions)
        mu_next_pred = self.dynamics(mu_t_sg, a_oh)   # seuls les poids de f ont des gradients

        dyn_loss = F.mse_loss(mu_next_pred, mu_next_sg)

        self.optim_dyn.zero_grad()
        dyn_loss.backward()
        self.optim_dyn.step()

        return {
            'recon_t': float(recon_t.detach().cpu()),
            'recon_next': float(recon_next.detach().cpu()),
            'recon_loss': float(recon_loss.detach().cpu()),
            'kl_loss': float(kl_loss.detach().cpu()),
            'cont_loss': float(cont_loss.detach().cpu()),
            'vae_loss': float(vae_loss.detach().cpu()),
            'dyn_loss': float(dyn_loss.detach().cpu())
        }

    # -------------------- phase wake (policy) --------------------
    def train_veil_step(self, obs, action, reward_ext, next_obs):
        """
        Mise à jour policy:
          - latent = µ (pas d'échantillon) + détaché
          - curiosité = MSE(µ_pred, µ_next) moyenne par dim, normalisée
          - bonus d'entropie
          - avantage simple: R_ext + beta * R_int_norm
        """
        obs      = self._ensure_batch_obs(obs)
        next_obs = self._ensure_batch_obs(next_obs)
        action   = torch.as_tensor([action], device=self.device, dtype=torch.long)

        with torch.no_grad():
            _, mu_t, _    = self._encode(obs,      sample=False)
            _, mu_next, _ = self._encode(next_obs, sample=False)
            mu_pred = self.predict_latent(mu_t, action)
            r_int   = F.mse_loss(mu_pred, mu_next, reduction='none').mean(dim=-1)  # (1,)

        # update stats curiosité
        self._update_rint_stats(r_int.item())
        r_int_n = self._rint_norm(r_int).detach()

        # policy sur µ_t détaché
        mu_det = mu_t.detach()
        logits  = self.policy(mu_det)
        logp    = F.log_softmax(logits, dim=-1).gather(1, action.view(-1, 1)).squeeze(1)
        entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(dim=-1)

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
