import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import Encoder, Decoder, LatentDynamics, Policy

class DeepMetaxisAgent:
    """
    Monde latent VAE (encoder/decoder + dynamics) + curiosité (erreur de dynamique réelle)
    + Critique extrinsèque latent V(μ)
    + Policy entraînée en imagination à partir de V(μ) et d'un terme de curiosité (ennui).

    Optimisations séparées:
      - VAE (reconstruction + KL + continuité temporelle)
      - Dynamique (prédiction de µ_{t+1} à partir de µ_t et a_t)
      - Critique extrinsèque latent
      - Policy (sur rollouts imaginés uniquement)
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
                 lr_value=1e-4,
                 weight_decay=0.0,
                 gamma_ext=0.99,
                 plan_horizon=10,
                 plan_tau=1.0,
                 lambda_plan=1.0):

        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Pondérations / hyperparams
        self.beta = beta                   # poids max curiosité (pour mode ennui)
        self.lambda_recon = lambda_recon   # poids reconstruction
        self.lambda_kl = lambda_kl         # poids KL VAE
        self.lambda_cont = lambda_cont     # poids continuité temporelle µ_t ~ µ_{t+1}
        self.ent_coef = ent_coef           # poids entropie (pour la policy si tu en rajoutes)
        self.gamma_ext = gamma_ext         # discount extrinsèque pour TD
        self.plan_horizon = plan_horizon   # N steps imaginés
        self.plan_tau = plan_tau           # température softmax(Q̃)
        self.lambda_plan = lambda_plan     # poids de la loss de planning sur la policy

        # --- Modules ---
        # Encoder "backbone" (features) -> µ/logvar
        self.encoder = Encoder(input_shape, latent_dim).to(device)
        self.fc_mu = nn.Linear(latent_dim, latent_dim).to(device)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim).to(device)

        self.decoder  = Decoder(latent_dim, input_shape).to(device)
        self.dynamics = LatentDynamics(latent_dim, action_dim).to(device)
        self.policy   = Policy(latent_dim, action_dim).to(device)

        # Critique extrinsèque latent
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

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
        # Critique extrinsèque
        self.optim_value = torch.optim.Adam(
            self.value_head.parameters(),
            lr=lr_value,
            weight_decay=weight_decay
        )
        # Policy (sur rollouts imaginés)
        self.optim_policy = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr_policy,
            weight_decay=weight_decay
        )

        # Stats online pour normaliser une curiosité basée sur erreur réelle (optionnel)
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

    # -------------------- interaction (act) --------------------
    def act(self, obs, greedy: bool = False):
        """
        Action dans le monde réel :
          - encode obs -> µ
          - policy(µ)
          - sample ou greedy
        """
        obs = self._ensure_batch_obs(obs)
        with torch.no_grad():
            _, mu, _ = self._encode(obs, sample=False)
            logits = self.policy(mu)
            if greedy:
                a = torch.argmax(logits, dim=-1)
            else:
                a = torch.distributions.Categorical(logits=logits).sample()
        return int(a.item())

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

    ## Fonction inutilisée
    def compute_intrinsic_reward_real(self, obs, action, next_obs):
        """
        Curiosité (erreur de dynamique) sur des données réelles (optionnelle pour logging / normalisation).
        obs, next_obs: (C,H,W) ou (B,C,H,W)
        action: scalaire ou (B,)
        """
        obs      = self._ensure_batch_obs(obs)
        next_obs = self._ensure_batch_obs(next_obs)
        action   = torch.as_tensor([action], device=self.device, dtype=torch.long)

        with torch.no_grad():
            _, mu_t, _    = self._encode(obs,      sample=False)
            _, mu_next, _ = self._encode(next_obs, sample=False)
            mu_pred = self.predict_latent(mu_t, action)
            err = F.mse_loss(mu_pred, mu_next, reduction='none').mean(dim=-1)  # (1,)
        return err.item()

    # -------------------- critique extrinsèque (réel) --------------------
    def _value(self, mu):
        """mu: (B,d) -> V(μ): (B,1)"""
        return self.value_head(mu)

    def _update_value_td(self, batch):
        """
        Mise à jour du critic extrinsèque latent (TD(0)).
        batch:
          'obs':      (B,C,H,W)
          'next_obs': (B,C,H,W)
          'rewards':  (B,)
          'dones':    (B,)  float ou bool
        """
        obs      = torch.as_tensor(batch['obs'],      device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device, dtype=torch.float32)
        rewards  = torch.as_tensor(batch['rewards'],  device=self.device, dtype=torch.float32).view(-1, 1)
        dones    = torch.as_tensor(batch['dones'],    device=self.device, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            _, mu_t, _    = self._encode(obs,      sample=False)
            _, mu_next, _ = self._encode(next_obs, sample=False)
            v_next = self._value(mu_next)
            target = rewards + self.gamma_ext * v_next * (1.0 - dones)

        v_pred = self._value(mu_t)

        loss = F.mse_loss(v_pred, target)

        self.optim_value.zero_grad()
        loss.backward()
        self.optim_value.step()

        return float(loss.detach().cpu())

    # -------------------- consolidation (sleep) --------------------
    def _update_consolidation(self, batch):
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
        z_t,    mu_t,    logvar_t    = self._encode(obs,      sample=True)
        z_next, mu_next, logvar_next = self._encode(next_obs, sample=True)

        x_hat_t    = self.decoder(z_t)
        x_hat_next = self.decoder(z_next)

        recon_t    = F.mse_loss(x_hat_t,    obs)
        recon_next = F.mse_loss(x_hat_next, next_obs)
        recon_loss = recon_t + recon_next

        kl_t    = self._kl_normal(mu_t,    logvar_t)
        kl_next = self._kl_normal(mu_next, logvar_next)
        kl_loss = 0.5 * (kl_t + kl_next)

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
        with torch.no_grad():
            _, mu_t_sg, _    = self._encode(obs,      sample=False)
            _, mu_next_sg, _ = self._encode(next_obs, sample=False)

        a_oh = self._one_hot(actions)
        mu_next_pred = self.dynamics(mu_t_sg, a_oh)

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

    def sleep_step(self, batch):
            """
            Sleep complet (monde + critic) à partir d'un batch :
            batch = {
                'obs': (B,C,H,W),
                'next_obs': (B,C,H,W),
                'actions': (B,),
                'rewards': (B,),
                'dones': (B,)
            }
            """
            # 1) monde latent (VAE + dynamique)
            world_batch = {
                "obs":      batch["obs"],
                "next_obs": batch["next_obs"],
                "actions":  batch["actions"],
            }
            logs_world = self._update_consolidation(world_batch)

            # 2) critic extrinsèque latent
            value_batch = {
                "obs":      batch["obs"],
                "next_obs": batch["next_obs"],
                "rewards":  batch["rewards"],
                "dones":    batch["dones"],
            }
            v_loss = self._update_value_td(value_batch)

            # on renvoie tout dans un seul dict
            logs_world["value_loss"] = v_loss
            return logs_world

    # -------------------- imagination & ennui --------------------

    # Fonction beta qui calculere beta pour gere la notion de curiosité
    def _beta_from_value(self, v_pred, v_thr=1.0, alpha=1.0):
        """
        v_pred: scalaire (valeur extrinsèque prédite à partir du rollout).
        Si v_pred << v_thr -> ennui -> beta_eff proche de beta.
        Si v_pred >> v_thr -> beta_eff ~ 0.
        """
        x = alpha * (v_thr - v_pred)
        sig = 1.0 / (1.0 + np.exp(-x))
        return float(self.beta * sig)

    def _rollout_imagined(self, mu_start, first_action, horizon=None):
        """
        Rollout latent en imagination:
          - premier pas forcé par first_action
          - ensuite actions échantillonnées depuis policy
        mu_start: (1,d)
        Retourne :
          mu_seq: (T,d) avec T = horizon+1
        """
        if horizon is None:
            horizon = self.plan_horizon

        mus = [mu_start]
        z = mu_start

        # step 0 -> 1 avec action imposée
        a_idx = torch.tensor([first_action], device=self.device, dtype=torch.long)
        z_next = self.predict_latent(z, a_idx)
        mus.append(z_next)
        z = z_next

        # steps suivants guidés par la policy
        for _ in range(horizon - 1):
            logits = self.policy(z)
            a_idx = torch.distributions.Categorical(logits=logits).sample()
            z_next = self.predict_latent(z, a_idx)
            mus.append(z_next)
            z = z_next

        return torch.cat(mus, dim=0)  # (T,d)

    # A faire :
    # Corriger experimentalement pour le calcul de beta
    def _imagined_q_for_actions(self, mu_t):
        """
        Pour chaque action a0, construit un rollout latent en imagination,
        evalue la "q-value" approximative:
          Q̃(a0) = V_ext(μ_T) + beta_eff * R_int_imag
        où R_int_imag est un proxy de curiosité (normes de déplacement latents).
        Retourne: tensor (A,) sur device.
        """
        q_vals = []
        for a0 in range(self.action_dim):
            with torch.no_grad():
                mu_seq = self._rollout_imagined(mu_t, first_action=a0)  # (T,d)
                mu_last = mu_seq[-1:, :]   # (1,d)

                # valeur extrinsèque prédite sur dernier état
                v_ext = self._value(mu_last)[0, 0].item()

                # proxy de curiosité : somme des ||μ_{t+1} - μ_t||
                diffs = mu_seq[1:] - mu_seq[:-1]  # (T-1,d)
                r_int_seq = torch.norm(diffs, dim=-1)  # (T-1,)
                r_int_tot = r_int_seq.sum().item()

                # ennui: plus V_ext est bas, plus on renforce curiosité
                beta_eff = self._beta_from_value(v_ext)

                q = v_ext + beta_eff * r_int_tot
            q_vals.append(q)

        return torch.tensor(q_vals, device=self.device, dtype=torch.float32)  # (A,)

    # -------------------- phase wake (policy, en imagination) --------------------
    def train_policy_imagination(self, obs):
        """
        Mise à jour de la policy uniquement en imagination:
          - encode obs -> µ_t
          - pour chaque action a0, rollout latent de horizon self.plan_horizon
          - estime Q̃(a0) à partir de V_ext + curiosité (ennui)
          - construit une distribution cible softmax(Q̃ / tau)
          - pousse la policy(µ_t) à se rapprocher de cette distribution (planning_loss)
        """
        obs = self._ensure_batch_obs(obs)  # (1,C,H,W)

        with torch.no_grad():
            _, mu_t, _ = self._encode(obs, sample=False)  # (1,d)
            q_imag = self._imagined_q_for_actions(mu_t)   # (A,)

            # distribution cible basée sur ces "Q imaginés"
            target_dist = torch.softmax(q_imag / self.plan_tau, dim=0)  # (A,)

        # distribution actuelle de la policy
        logits = self.policy(mu_t)      # (1,A)
        pi = torch.softmax(logits, dim=-1).squeeze(0)  # (A,)

        # KL(target || pi) ~ cross-entropy
        # Ajuster la politique pour se rapprocher de la distribution cible
        planning_loss = torch.sum(
            target_dist * (torch.log(target_dist + 1e-8) - torch.log(pi + 1e-8))
        )

        # (optionnel) bonus d'entropie si tu veux
        entropy = -(pi * torch.log(pi + 1e-8)).sum()

        loss = self.lambda_plan * planning_loss - self.ent_coef * entropy

        self.optim_policy.zero_grad()
        loss.backward()
        self.optim_policy.step()

        return {
            "planning_loss": float(planning_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu())
        }
