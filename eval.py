import crafter
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from model import WorldModel, ActorCritic
import os
import imageio
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_path="world_model_final.safetensors", actor_path="actor_critic_final.safetensors", episodes=3):
    env = crafter.Env(seed=42)
    os.makedirs('./eval_recordings', exist_ok=True)
    
    action_dim = env.action_space.n
    
    print(f"Loading World Model from {model_path}...")
    world_model = WorldModel(action_dim=action_dim).to(device)
    world_model.load_state_dict(load_file(model_path))
    world_model.eval()
    
    print(f"Loading Actor-Critic from {actor_path}...")
    actor_critic = ActorCritic(feat_dim=world_model.feat_dim, action_dim=action_dim).to(device)
    actor_critic.load_state_dict(load_file(actor_path))
    actor_critic.eval()
    
    print(f"\nStarting evaluation for {episodes} episodes...")
    for ep in range(episodes):
        obs = env.reset()
        frames = [obs]
        h, z = world_model.rssm.initial_state(batch_size=1, device=device)
        
        done = False
        total_reward = 0
        step = 0
        action_counts = Counter()
        
        while not done:
            with torch.no_grad():
                # Process observation
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                enc_out = world_model.encoder(obs_tensor)
                
                # Get posterior state using the observation
                z_post, _ = world_model.rssm.step_posterior(h, enc_out)
                feat = torch.cat([h, z_post], dim=-1)
                
                # Select action (explore=True samples from the distribution since the model is undertrained)
                action = actor_critic.select_action(feat, explore=True).item()
                action_counts[action] += 1
            
            # Step environment
            obs, reward, done, info = env.step(action)
            frames.append(obs)
            total_reward += reward
            step += 1
            
            # Advance prior state based on the action taken
            act_onehot = F.one_hot(torch.tensor([action], device=device), num_classes=action_dim).float()
            h, z, _ = world_model.rssm.step_prior(h, z_post, act_onehot)
            
        print(f"Episode {ep + 1} finished in {step} steps. Total Reward: {total_reward:.1f}")
        
        # Print a breakdown of the actions it took!
        print("Action Breakdown:")
        for act, count in action_counts.most_common(5):
            print(f"  Action {act}: {count} times")
            
        # Save video as GIF to completely bypass all PyAV codec bugs
        video_path = f"./eval_recordings/eval_ep{ep+1}.gif"
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Saved video to {video_path}")
        
    print("\nEvaluation complete!")

if __name__ == "__main__":
    # Note: Update these filenames if your training stopped early (e.g., "world_model_20000.safetensors")
    evaluate("world_model_final.safetensors", "actor_critic_final.safetensors")
