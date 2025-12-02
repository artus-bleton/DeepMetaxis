# run_deep_metaxis_crafter.py
import os
import json
import time
import random
from collections import defaultdict, deque

import numpy as np
import torch

# --- Compat NumPy 2.x pour l'écosystème Gym ancien (Crafter 1.x) ---
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import crafter

from modules import Encoder, Decoder, LatentDynamics, Policy
from DeepMetaxisAgent import DeepMetaxisAgent


# ---------------------- Utils ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    if obs.dtype != np.float32:
        obs = obs.astype(np.float32) / 255.0
    return np.transpose(obs, (2, 0, 1))  # CHW


CANONICAL_ACH = [
    'collect_coal','collect_diamond','collect_drink','collect_iron',
    'collect_stone','collect_wood','defeat_skeleton','defeat_zombie',
    'eat_plant','eat_cow','make_iron_pickaxe','make_stone_pickaxe',
    'make_wood_pickaxe','make_furnace','make_table','place_furnace',
    'place_plant','place_rock','place_table','sleep','wake_up','collect_sapling'
]


# ---------------------- Metrics ----------------------
class CrafterMetrics:
    """
    - Success rates par achievement: P(achievement obtenu ≥1 fois par épisode)
    - Score = 100 * (prod_i rate_i)^(1/N), avec N=22 (fixe).
    """
    def __init__(self, all_achievements=None):
        if isinstance(all_achievements, (list, tuple)) and len(all_achievements) == 22:
            self.all_achievements = list(all_achievements)
        else:
            self.all_achievements = list(CANONICAL_ACH)

        self.episodes = 0
        self.ep_seen = set()
        self.success_counts = defaultdict(int)

    def start_episode(self):
        self.ep_seen = set()

    def step_info(self, info: dict):
        ach = info.get('achievement', None)
        if isinstance(ach, str):
            self.ep_seen.add(ach)
        achs = info.get('achievements', None)
        if isinstance(achs, dict):
            for k, v in achs.items():
                if v:
                    self.ep_seen.add(k)

    def end_episode(self):
        self.episodes += 1
        for a in self.ep_seen:
            if a in self.all_achievements:
                self.success_counts[a] += 1

    def rates(self):
        if self.episodes == 0:
            return {k: 0.0 for k in self.all_achievements}
        return {k: self.success_counts.get(k, 0) / self.episodes
                for k in self.all_achievements}

    def score(self):
        rs = self.rates()
        vals = [float(rs[k]) for k in self.all_achievements]
        if not vals:
            return 0.0
        prod = 1.0
        for r in vals:
            prod *= max(0.0, float(r))
            if prod == 0.0:
                return 0.0
        geom = prod ** (1.0 / len(vals))
        return 100.0 * geom

    def summary(self):
        rs = self.rates()
        return {
            "episodes": int(self.episodes),
            "num_achievements": int(len(self.all_achievements)),
            "success_rates": {k: float(rs[k]) for k in self.all_achievements},
            "crafter_score_percent": float(self.score()),
        }


# ---------------------- Prioritized Replay ----------------------
class PrioritizedReplayBuffer:
    """
    Prioritisation simple (proportionnelle) sur un score fourni à l'add (p.ex. curiosité/dyn-err).
    Pas d'IS weights (facultatif ici). Clamp des priorités pour éviter la dégénérescence.
    """
    def __init__(self, capacity=100_000, alpha=0.6, p_min=1e-3, p_max=10.0):
        self.capacity = capacity
        self.alpha = alpha
        self.p_min = p_min
        self.p_max = p_max

        self.obs, self.next_obs, self.actions = [], [], []
        self.priorities = []

    def add(self, o, no, a, priority: float):
        if len(self.obs) >= self.capacity:
            self.obs.pop(0); self.next_obs.pop(0); self.actions.pop(0); self.priorities.pop(0)
        p = float(np.clip(priority, self.p_min, self.p_max))
        self.obs.append(o); self.next_obs.append(no); self.actions.append(a); self.priorities.append(p)

    def sample_batch(self, batch_size=64):
        if len(self.obs) < batch_size:
            raise ValueError("Not enough samples in buffer.")
        ps = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        ps /= ps.sum()
        idx = np.random.choice(len(self.obs), size=batch_size, replace=False, p=ps)
        return {
            "obs": np.stack([self.obs[i] for i in idx]),
            "next_obs": np.stack([self.next_obs[i] for i in idx]),
            "actions": np.array([self.actions[i] for i in idx], dtype=np.int64),
        }

    def __len__(self):
        return len(self.obs)


# ---------------------- Evaluation ----------------------
def eval_agent(make_env_fn, agent, n_episodes=20, size=64, reward=True, seed=0):
    env = make_env_fn(size=size, reward=reward, seed=seed)
    ach_list = None
    for attr in ("achievements", "_achievements", "achievement_names", "_achievement_names"):
        if hasattr(env, attr):
            maybe = getattr(env, attr)
            if isinstance(maybe, (list, tuple)) and len(maybe) == 22:
                ach_list = list(maybe)
                break
            if isinstance(maybe, dict) and len(maybe) == 22:
                ach_list = list(maybe.keys())
                break
    metrics = CrafterMetrics(all_achievements=ach_list)

    for _ in range(n_episodes):
        obs = env.reset()
        metrics.start_episode()
        done = False
        while not done:
            o = preprocess_obs(obs)
            try:
                a = agent.act(o, greedy=True)
            except TypeError:
                a = agent.act(o)
            obs, _, done, info = env.step(a)
            metrics.step_info(info)
        metrics.end_episode()

    env.close()
    s = metrics.summary()
    assert s["num_achievements"] == 22, f"num_achievements={s['num_achievements']}, attendu=22"
    return s


# ---------------------- Main ----------------------
def main():
    # ---------------- HParams ----------------
    seed = 0
    size = 64
    reward_flag = True
    latent_dim = 128
    beta_target = 0.1            # β final (curiosité)
    beta_warm = 0.02             # β de départ
    beta_anneal_steps = 50_000   # annealing linéaire vers beta_target
    lambda_recon = 1.0

    batch_size = 64
    sleep_every = 32             # tentative initiale (sera adapté)
    sleep_updates_per_tick = 4   # nb d'updates sommeil quand déclenché
    sleep_updates_min = 1
    sleep_updates_max = 16

    warmup_steps = 100        # steps de collecte random + consolidation massive
    max_episodes = 1_000_000
    max_steps = 10_000
    max_interactions = 1_000_000

    eval_every = 10_000
    eval_episodes = 20

    # Adaptation automatique selon la dynamique
    dyn_ma_window = 200
    dyn_up_factor = 2.0          # si explosion, on multiplie sleep_updates_per_tick (borné)
    dyn_down_factor = 0.85       # si stable longtemps, on réduit doucement
    dyn_hi_quantile = 0.8
    dyn_lo_quantile = 0.3

    set_seed(seed)

    def make_env(size=64, reward=True, seed=None):
        env = crafter.Env(size=size, reward=reward)
        if seed is not None:
            try: env.seed(seed)
            except Exception: pass
        return env

    env = make_env(size=size, reward=reward_flag, seed=seed)
    obs = env.reset()

    o0 = preprocess_obs(obs)
    input_shape = o0.shape
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DeepMetaxisAgent(
        input_shape=input_shape,
        latent_dim=latent_dim,
        action_dim=action_dim,
        beta=beta_warm,
        lambda_recon=lambda_recon,
        device=device
    )

    # Liste d'achievements depuis env sinon canonique
    all_achievements = None
    for attr in ("achievements", "_achievements", "achievement_names", "_achievement_names"):
        if hasattr(env, attr):
            maybe = getattr(env, attr)
            if isinstance(maybe, (list, tuple)) and len(maybe) == 22:
                all_achievements = list(maybe); break
            if isinstance(maybe, dict) and len(maybe) == 22:
                all_achievements = list(maybe.keys()); break

    metrics = CrafterMetrics(all_achievements=all_achievements)

    rb = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)

    global_step = 0
    dyn_loss_hist = deque(maxlen=dyn_ma_window)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    def hparam_stamp():
        return (
            f"size{size}_reward{int(reward_flag)}_betaT{beta_target}_lam{lambda_recon}"
            f"_lat{latent_dim}_bs{batch_size}_sleep{sleep_every}"
            f"_warm{warmup_steps}_evalEvery{eval_every}_evalEp{eval_episodes}_seed{seed}"
        )

    # ---------- Phase WARMUP : random policy + consolidation dense ----------
    print(f"[warmup] collecting {warmup_steps} random steps...")
    env_w = make_env(size=size, reward=reward_flag, seed=seed+1)
    w_obs = env_w.reset()
    while global_step < warmup_steps:
        o = preprocess_obs(w_obs)
        a = env_w.action_space.sample()
        next_obs, rew, done, _ = env_w.step(a)
        no = preprocess_obs(next_obs)

        # priorité = proxy curiosité instantanée (erreur de dynamique) sans gradient
        with torch.no_grad():
            to  = torch.as_tensor(o,  device=device, dtype=torch.float32).unsqueeze(0)
            tno = torch.as_tensor(no, device=device, dtype=torch.float32).unsqueeze(0)
            z  = agent.encoder(to)
            zn = agent.encoder(tno)
            z_pred = agent.predict_latent(z, torch.tensor([a], device=device))
            priority = torch.mean((z_pred - zn).pow(2), dim=-1).item()
        rb.add(o, no, int(a), priority)

        # consolidation dense pendant warmup si possible
        if len(rb) >= batch_size:
            for _ in range(8):  # updates plus nombreux au warmup
                batch = rb.sample_batch(batch_size=batch_size)
                sleep_logs = agent.update_consolidation(batch)
                dyn_loss_hist.append(sleep_logs['dyn_loss'])

        w_obs = next_obs
        global_step += 1
        if done:
            w_obs = env_w.reset()
    env_w.close()
    print("[warmup] done.")

    # ---------- Boucle principale ----------
    for ep in range(max_episodes):
        if global_step >= max_interactions:
            break
        obs = env.reset()
        metrics.start_episode()
        ep_return = 0.0

        for t in range(max_steps):
            if global_step >= max_interactions:
                break

            # Annealing de β (curiosité)
            if global_step < beta_anneal_steps:
                agent.beta = beta_warm + (beta_target - beta_warm) * (global_step / beta_anneal_steps)
            else:
                agent.beta = beta_target

            # Act / step env
            o = preprocess_obs(obs)
            action = agent.act(o)
            next_obs, reward, done, info = env.step(action)
            no = preprocess_obs(next_obs)
            ep_return += float(reward)
            metrics.step_info(info)

            # Policy update (wake)
            agent.train_veil_step(o, action, float(reward), no)

            # Priorité = proxy curiosité locale
            with torch.no_grad():
                to  = torch.as_tensor(o,  device=device, dtype=torch.float32).unsqueeze(0)
                tno = torch.as_tensor(no, device=device, dtype=torch.float32).unsqueeze(0)
                z  = agent.encoder(to)
                zn = agent.encoder(tno)
                z_pred = agent.predict_latent(z, torch.tensor([action], device=device))
                priority = torch.mean((z_pred - zn).pow(2), dim=-1).item()

            rb.add(o, no, int(action), priority)
            global_step += 1

            # --- Consolidation planifiée + adaptative ---
            if len(rb) >= batch_size and global_step % sleep_every == 0:
                # nombre d'updates sommeil adaptatif (borné)
                updates = sleep_updates_per_tick
                for _ in range(updates):
                    batch = rb.sample_batch(batch_size=batch_size)
                    sleep_logs = agent.update_consolidation(batch)
                    dyn_loss_hist.append(sleep_logs['dyn_loss'])
                # adaptation via quantiles de la fenêtre glissante
                if len(dyn_loss_hist) >= max(32, dyn_ma_window // 4):
                    losses = np.array(dyn_loss_hist, dtype=np.float32)
                    q_hi = float(np.quantile(losses, dyn_hi_quantile))
                    q_lo = float(np.quantile(losses, dyn_lo_quantile))
                    # si forte -> on augmente les updates sommeil
                    if q_hi > 2.0 * (q_lo + 1e-6):
                        sleep_updates_per_tick = min(int(sleep_updates_per_tick * dyn_up_factor), sleep_updates_max)
                    # si très stable -> on réduit légèrement
                    elif q_hi < 1.2 * (q_lo + 1e-6):
                        sleep_updates_per_tick = max(int(max(sleep_updates_per_tick * dyn_down_factor, sleep_updates_min)), sleep_updates_min)

                print(f"[sleep] step={global_step} dyn={sleep_logs['dyn_loss']:.4f} "
                      f"recon={sleep_logs['recon_loss']:.4f} updates={updates} β={agent.beta:.4f}")

            # Évaluation périodique (greedy)
            if global_step % eval_every == 0:
                es = eval_agent(make_env, agent, n_episodes=eval_episodes,
                                size=size, reward=reward_flag, seed=seed+123)
                stamp = hparam_stamp()
                ts = int(time.time())
                out_path = f"logs/crafter_eval_{stamp}_{ts}.json"
                payload = {
                    "global_step": global_step,
                    "episodes_so_far": metrics.episodes,
                    "train_summary": metrics.summary(),
                    "eval_summary": es,
                    "hparams": {
                        "seed": seed, "size": size, "reward": reward_flag,
                        "latent_dim": latent_dim, "beta_target": beta_target,
                        "beta_warm": beta_warm, "lambda_recon": lambda_recon,
                        "batch_size": batch_size, "sleep_every": sleep_every,
                        "sleep_updates_per_tick": sleep_updates_per_tick,
                        "warmup_steps": warmup_steps,
                        "max_interactions": max_interactions, "eval_every": eval_every,
                        "eval_episodes": eval_episodes, "device": device,
                    }
                }
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"[EVAL] step={global_step} score={es['crafter_score_percent']:.2f}% -> {out_path}")

            obs = next_obs
            if done:
                break

        metrics.end_episode()
        print(f"[episode {ep}] return={ep_return:.2f}, steps={t+1}, sleep_updates/tick={sleep_updates_per_tick}")

    env.close()

    # Sauvegarde checkpoint
    ckpt_path = f"checkpoints/deep_metaxis_crafter_{hparam_stamp()}.pt"
    torch.save({
        "encoder": agent.encoder.state_dict(),
        "decoder": agent.decoder.state_dict(),
        "dynamics": agent.dynamics.state_dict(),
        "policy": agent.policy.state_dict(),
    }, ckpt_path)
    print(f"Modèle sauvegardé -> {ckpt_path}")

    # Sauvegarde métriques finales (train online)
    final_path = f"logs/crafter_metrics_{hparam_stamp()}_{int(time.time())}.json"
    with open(final_path, "w") as f:
        json.dump(metrics.summary(), f, indent=2)
    print("Métriques (train online) sauvegardées ->", final_path)


if __name__ == "__main__":
    main()
