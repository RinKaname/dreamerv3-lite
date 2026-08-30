import crafter
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file, load_file
import os
import glob
import torch.nn.functional as F
import torch.distributions as D

from model import WorldModel, ActorCritic, symlog, symexp, twohot_loss, twohot_decode

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 100_000
TRAIN_EVERY = 5          
TRAIN_STEPS = 1          
PREFILL_STEPS = 100     
BATCH_SIZE = 4
SEQ_LEN = 50
IMAGINE_HORIZON = 15

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.obs = np.empty((capacity, 64, 64, 3), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_sequence(self, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        start_indices = []
        attempts = 0
        max_attempts = batch_size * 20
        
        while len(start_indices) < batch_size and attempts < max_attempts:
            i = np.random.randint(0, max(1, self.size - seq_len))
            if not np.any(self.dones[i : i + seq_len - 1]):
                start_indices.append(i)
            attempts += 1
            
        if len(start_indices) < batch_size:
            # Fallback if buffer is too fragmented
            start_indices = list(np.random.randint(0, max(1, self.size - seq_len), size=batch_size))
            
        obs_seq = np.stack([self.obs[i : i + seq_len] for i in start_indices])
        act_seq = np.stack([self.actions[i : i + seq_len] for i in start_indices])
        rew_seq = np.stack([self.rewards[i : i + seq_len] for i in start_indices])
        don_seq = np.stack([self.dones[i : i + seq_len] for i in start_indices])
        
        # Keep obs in [0, 255] range for the Encoder's symlog
        obs_tensor = torch.tensor(obs_seq, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
        act_tensor = torch.tensor(act_seq, dtype=torch.int64, device=device)
        rew_tensor = torch.tensor(rew_seq, dtype=torch.float32, device=device).unsqueeze(-1)
        cont_tensor = 1.0 - torch.tensor(don_seq, dtype=torch.float32, device=device)
        
        return obs_tensor, act_tensor, rew_tensor, cont_tensor

def compute_lambda_returns(rewards, continues, values, lambda_=0.95, gamma=0.99):
    """Computes generalized lambda returns for Actor-Critic."""
    H, B = rewards.shape[:2]
    returns = torch.zeros_like(values)
    
    # Bootstrap from the last value
    last_val = values[-1]
    
    for t in reversed(range(H)):
        next_val = values[t + 1] if t + 1 < H else values[-1]
        returns[t] = rewards[t] + continues[t] * gamma * ((1 - lambda_) * next_val + lambda_ * last_val)
        last_val = returns[t]
    return returns

def train():
    env = crafter.Env(seed=0)
    action_dim = env.action_space.n
    
    world_model = WorldModel(action_dim=action_dim).to(device)
    actor_critic = ActorCritic(feat_dim=world_model.feat_dim, action_dim=action_dim).to(device)
    
    # --- Resume Logic ---
    wm_path = "world_model_final.safetensors"
    ac_path = "actor_critic_final.safetensors"
    if os.path.exists(wm_path) and os.path.exists(ac_path):
        print(f"Found saved weights! Resuming training from {wm_path} and {ac_path}...")
        world_model.load_state_dict(load_file(wm_path))
        actor_critic.load_state_dict(load_file(ac_path))
    else:
        print("No saved weights found. Starting training from scratch...")
    
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=1e-4, eps=1e-8)
    ac_opt = torch.optim.Adam(actor_critic.parameters(), lr=3e-5, eps=1e-5)
    
    buffer = ReplayBuffer(capacity=100_000)
    
    # --- Load Human Demonstrations (Imitation Learning) ---
    human_dir = "./human_data"
    if os.path.exists(human_dir):
        files = glob.glob(f"{human_dir}/*.npz")
        if files:
            print(f"Loading {len(files)} human demonstrations into the brain...")
            for f in files:
                data = np.load(f, allow_pickle=True)
                for obs, act, rew, done in zip(data['image'], data['action'], data['reward'], data['done']):
                    buffer.add(obs, act, rew, done)
            print(f"Matrix upload complete! The AI now has {buffer.size} frames of human experience.")
    
    obs = env.reset()
    h, z = world_model.rssm.initial_state(batch_size=1, device=device)
    
    print("Prefilling buffer...")
    for _ in tqdm(range(PREFILL_STEPS), desc="Prefill"):
        act = env.action_space.sample()
        next_obs, rew, done, _ = env.step(act)
        buffer.add(obs, act, rew, done)
        obs = env.reset() if done else next_obs

    print("Starting Training...")
    pbar = tqdm(range(TOTAL_STEPS), desc="Training Steps")
    
    for step in pbar:
        # 1. Environment Interaction
        with torch.no_grad():
            if True:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                
                enc_out = world_model.encoder(obs_tensor)
                z_post, _ = world_model.rssm.step_posterior(h, enc_out)
                feat = torch.cat([h, z_post], dim=-1)
                
                action = actor_critic.select_action(feat, explore=True).item()

        next_obs, rew, done, info = env.step(action)
        
        # Dense reward shaping for early game
        shaped_rew = rew
        if isinstance(info, dict) and 'achievements' in info:
            ach = info['achievements']
            if ach.get('collect_wood', 0) > 0: shaped_rew += 2.0
            if ach.get('place_table', 0) > 0: shaped_rew += 5.0
            if ach.get('collect_stone', 0) > 0: shaped_rew += 2.0
            
        buffer.add(obs, action, shaped_rew, done)
        
        if done:
            obs = env.reset()
            h, z = world_model.rssm.initial_state(batch_size=1, device=device)
        else:
            obs = next_obs
            # Advance RNN state via Prior for next step acting
            act_onehot = F.one_hot(torch.tensor([action], device=device), num_classes=action_dim).float()
            h, z, _ = world_model.rssm.step_prior(h, z_post, act_onehot)

        # 2. Periodic Network Updates
        if step % TRAIN_EVERY == 0 and buffer.size >= PREFILL_STEPS:
            for _ in range(TRAIN_STEPS):
                b_obs, b_act, b_rew, b_cont = buffer.sample_sequence()
                
                # --- Step A: Train World Model ---
                wm_opt.zero_grad()
                if True:
                    # Unroll RSSM
                    post_states, prior_logits, post_logits = world_model.unroll(b_obs, b_act, b_cont)
                    
                    # Losses
                    rec_obs = world_model.decoder(post_states)
                    rec_loss = F.mse_loss(rec_obs, symlog(b_obs))
                    
                    rew_preds = world_model.reward_predictor(post_states)
                    rew_loss = twohot_loss(rew_preds, symlog(b_rew.squeeze(-1)))
                    
                    cont_preds = world_model.continue_predictor(post_states)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_preds.squeeze(-1), b_cont)
                    
                    kl_loss = world_model.kl_loss(post_logits, prior_logits)
                    
                    wm_total_loss = rec_loss + rew_loss + cont_loss + 0.5 * kl_loss

                wm_total_loss.backward()
                nn.utils.clip_grad_norm_(world_model.parameters(), 1000.0)
                wm_opt.step()
                
                # --- Step B: Train Actor-Critic ---
                ac_opt.zero_grad()
                if True:
                    initial_states = post_states.detach().view(-1, world_model.feat_dim)
                    imag_feats, imag_actions, next_feats = world_model.imagine(initial_states, actor_critic, horizon=IMAGINE_HORIZON)
                    
                    imag_rew_logits = world_model.reward_predictor(next_feats)
                    imag_rewards = symexp(twohot_decode(imag_rew_logits)).squeeze(-1)
                    
                    imag_cont_logits = world_model.continue_predictor(next_feats)
                    imag_cont = torch.sigmoid(imag_cont_logits).squeeze(-1)
                    
                    val_logits = actor_critic.ema_critic(next_feats)
                    next_values = symexp(twohot_decode(val_logits)).squeeze(-1)
                    
                    returns = compute_lambda_returns(imag_rewards, imag_cont, next_values)
                    
                    policy_logits = actor_critic.actor(imag_feats.detach())
                    policy_dist = D.Categorical(logits=policy_logits)
                    log_probs = policy_dist.log_prob(imag_actions)
                    
                    baseline_logits = actor_critic.critic(imag_feats.detach())
                    baseline = symexp(twohot_decode(baseline_logits)).squeeze(-1)
                    
                    ret_percentiles = torch.quantile(returns.detach(), torch.tensor([0.05, 0.95], device=device))
                    ret_scale = torch.clamp(ret_percentiles[1] - ret_percentiles[0], min=1.0)
                    
                    advantage = returns.detach() - baseline.detach()
                    actor_loss = -torch.mean(log_probs * (advantage / ret_scale))
                    
                    entropy_loss = -3e-3 * policy_dist.entropy().mean()
                    
                    critic_loss = twohot_loss(baseline_logits, symlog(returns.detach()))
                    
                    ac_total_loss = actor_loss + entropy_loss + critic_loss
                    
                ac_total_loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), 100.0)
                ac_opt.step()
                
                actor_critic.update_ema()
                
                # Clear cache to prevent Windows WDDM fragmentation crashes
                torch.cuda.empty_cache()

            if step % 500 == 0:
                pbar.set_postfix({
                    "WM_L": f"{wm_total_loss.item():.2f}",
                    "AC_L": f"{ac_total_loss.item():.2f}",
                    "VRAM": f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB"
                })
            
            if step > 0 and step % 10000 == 0:
                save_file(world_model.state_dict(), f"world_model_{step}.safetensors")
                save_file(actor_critic.state_dict(), f"actor_critic_{step}.safetensors")

    save_file(world_model.state_dict(), "world_model_final.safetensors")
    save_file(actor_critic.state_dict(), "actor_critic_final.safetensors")
    print("Training complete! Models saved.")

if __name__ == "__main__":
    train()