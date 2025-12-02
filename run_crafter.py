# run_deep_metaxis_crafter_tb.py
import random
import numpy as np
import torch
import crafter
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from DeepMetaxisAgent import DeepMetaxisAgent


# ---------- Utils ----------
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


# ---------- Metrics ----------
class CrafterMetrics:
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

    def step_info(self, info):
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


# ---------- Replay Buffer étendu ----------
class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.capacity = capacity
        self.obs, self.next_obs = [], []
        self.actions, self.rewards, self.dones = [], [], []

    def add(self, o, no, a, r, d):
        if len(self.obs) >= self.capacity:
            self.obs.pop(0)
            self.next_obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
        self.obs.append(o)
        self.next_obs.append(no)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(d)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.obs), size=batch_size, replace=False)
        return {
            "obs":      np.stack([self.obs[i]      for i in idx]),
            "next_obs": np.stack([self.next_obs[i] for i in idx]),
            "actions":  np.array([self.actions[i]  for i in idx], dtype=np.int64),
            "rewards":  np.array([self.rewards[i]  for i in idx], dtype=np.float32),
            "dones":    np.array([self.dones[i]    for i in idx], dtype=np.float32),
        }

    def __len__(self):
        return len(self.obs)


# ---------- Main ----------
def main():
    set_seed(0)

    size = 64
    latent_dim = 128
    sleep_every = 32
    batch_size = 64
    max_steps = 5000
    max_interactions = 200_000

    env = crafter.Env(size=size, reward=True)
    obs = env.reset()
    input_shape = preprocess_obs(obs).shape
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DeepMetaxisAgent(
        input_shape, latent_dim, action_dim,
        beta=0.1, lambda_recon=1.0, device=device
    )

    rb = ReplayBuffer()
    metrics = CrafterMetrics()
    global_step = 0

    # --- TensorBoard writer ---
    writer = SummaryWriter(log_dir="runs/deep_metaxis_crafter_imagination")

    print("Training... (lance `tensorboard --logdir runs` pour monitorer)")

    for ep in range(999999):
        if global_step >= max_interactions:
            break

        obs = env.reset()
        metrics.start_episode()
        ep_return = 0.0

        sleep_dyn = []
        sleep_rec = []
        value_losses = []
        plan_losses = []

        for step in range(max_steps):
            o = preprocess_obs(obs)

            # action dans le monde réel
            a = agent.act(o)
            next_obs, rew, done, info = env.step(a)
            no = preprocess_obs(next_obs)

            ep_return += float(rew)
            metrics.step_info(info)

            # buffer réel (pour monde + critic)
            rb.add(o, no, int(a), float(rew), float(done))
            global_step += 1

            # --- update policy en imagination à partir de l'état courant ---
            plan_logs = agent.train_policy_imagination(o)
            plan_losses.append(plan_logs["planning_loss"])

            # --- consolidation + critic sur batch périodique ---
            if len(rb) >= batch_size and global_step % sleep_every == 0:
                batch = rb.sample(batch_size)

                world_batch = {
                    "obs":      batch["obs"],
                    "next_obs": batch["next_obs"],
                    "actions":  batch["actions"],
                }
                logs_world = agent.update_consolidation(world_batch)
                sleep_dyn.append(logs_world['dyn_loss'])
                sleep_rec.append(logs_world['recon_loss'])

                value_batch = {
                    "obs":      batch["obs"],
                    "next_obs": batch["next_obs"],
                    "rewards":  batch["rewards"],
                    "dones":    batch["dones"],
                }
                v_loss = agent.update_value_td(value_batch)
                value_losses.append(v_loss)

            obs = next_obs
            if done or global_step >= max_interactions:
                break

        metrics.end_episode()

        # stats épisode
        ach = len(metrics.ep_seen)
        total_ach = len(metrics.all_achievements)
        ach_ratio = ach / total_ach if total_ach > 0 else 0.0

        mean_dyn = float(np.mean(sleep_dyn))     if sleep_dyn   else 0.0
        mean_rec = float(np.mean(sleep_rec))     if sleep_rec   else 0.0
        mean_v   = float(np.mean(value_losses))  if value_losses else 0.0
        mean_pl  = float(np.mean(plan_losses))   if plan_losses  else 0.0

        print(
            f"[EP {ep}] return={ep_return:.2f} | dyn={mean_dyn:.4f} "
            f"| recon={mean_rec:.4f} | V_loss={mean_v:.4f} "
            f"| plan_loss={mean_pl:.4f} | achievements {ach}/{total_ach} ({ach_ratio:.2f})"
        )

        # ----- Log TensorBoard -----
        writer.add_scalar("episode/return", ep_return, ep)
        writer.add_scalar("episode/mean_dyn_loss", mean_dyn, ep)
        writer.add_scalar("episode/mean_recon_loss", mean_rec, ep)
        writer.add_scalar("episode/mean_value_loss", mean_v, ep)
        writer.add_scalar("episode/mean_planning_loss", mean_pl, ep)
        writer.add_scalar("episode/achievements_ratio", ach_ratio, ep)

    env.close()
    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
