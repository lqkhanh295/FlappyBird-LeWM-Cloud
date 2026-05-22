import os
import sys
import h5py
import numpy as np
from pathlib import Path

# Add current dir to path to ensure src can be imported
sys.path.append(str(Path(__file__).parent))
from src.flappy_env import FlappyEnv

def get_expert_action(obs, info):
    state = info["state"]
    pipe_dist_x = state[0]
    pipe_gap_center_y = state[1]
    bird_y = state[2]
    
    vel_y = info["proprio"][1]
    
    # Expert rule: flap if bird is too low relative to the next pipe gap center
    # Target height is slightly above the gap center to compensate for gravity delay
    target_y = pipe_gap_center_y - 10
    
    if bird_y > target_y:
        if vel_y > -2:  # Avoid flapping if already moving upwards quickly
            return 1
    return 0

def collect_dataset(num_episodes=50, output_path=None):
    env = FlappyEnv(mute=True)
    
    print(f"Starting data collection for {num_episodes} episodes...")
    print(f"Saving dataset to {output_path}...", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create extendable datasets
        ds_pixels = f.create_dataset('pixels', shape=(0, 224, 224, 3), maxshape=(None, 224, 224, 3), dtype=np.uint8, compression='gzip')
        ds_action = f.create_dataset('action', shape=(0, 1), maxshape=(None, 1), dtype=np.float32, compression='gzip')
        ds_proprio = f.create_dataset('proprio', shape=(0, 2), maxshape=(None, 2), dtype=np.float32, compression='gzip')
        ds_state = f.create_dataset('state', shape=(0, 3), maxshape=(None, 3), dtype=np.float32, compression='gzip')
        ds_episode_idx = f.create_dataset('episode_idx', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip')
        ds_step_idx = f.create_dataset('step_idx', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip')
        
        total_steps = 0
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0
            
            ep_pixels = [obs]
            ep_actions = []
            ep_proprios = [info["proprio"]]
            ep_states = [info["state"]]
            ep_episode_idx = [ep]
            ep_step_idx = [step]
            
            while not done:
                action = get_expert_action(obs, info)
                obs, reward, done, info = env.step(action)
                
                ep_pixels.append(obs)
                ep_actions.append(action)
                ep_proprios.append(info["proprio"])
                ep_states.append(info["state"])
                step += 1
                ep_episode_idx.append(ep)
                ep_step_idx.append(step)
                
                if step >= 300:  # Cap episode length to save disk space
                    break
                    
            # Pad the final action to match observation length
            ep_actions.append(0)
            
            # Convert to numpy
            ep_pixels_np = np.array(ep_pixels, dtype=np.uint8)
            ep_actions_np = np.array(ep_actions, dtype=np.float32).reshape(-1, 1)
            ep_proprios_np = np.array(ep_proprios, dtype=np.float32)
            ep_states_np = np.array(ep_states, dtype=np.float32)
            ep_episode_idx_np = np.array(ep_episode_idx, dtype=np.int32)
            ep_step_idx_np = np.array(ep_step_idx, dtype=np.int32)
            
            num_transitions = len(ep_pixels_np)
            
            # Append to HDF5
            ds_pixels.resize(ds_pixels.shape[0] + num_transitions, axis=0)
            ds_pixels[-num_transitions:] = ep_pixels_np
            
            ds_action.resize(ds_action.shape[0] + num_transitions, axis=0)
            ds_action[-num_transitions:] = ep_actions_np
            
            ds_proprio.resize(ds_proprio.shape[0] + num_transitions, axis=0)
            ds_proprio[-num_transitions:] = ep_proprios_np
            
            ds_state.resize(ds_state.shape[0] + num_transitions, axis=0)
            ds_state[-num_transitions:] = ep_states_np
            
            ds_episode_idx.resize(ds_episode_idx.shape[0] + num_transitions, axis=0)
            ds_episode_idx[-num_transitions:] = ep_episode_idx_np
            
            ds_step_idx.resize(ds_step_idx.shape[0] + num_transitions, axis=0)
            ds_step_idx[-num_transitions:] = ep_step_idx_np
            
            total_steps += num_transitions
            print(f"Episode {ep+1}/{num_episodes} finished. Steps: {step}, Total Steps: {total_steps}, Score: {info['score']}", flush=True)
            
    env.close()
    print(f"Dataset saved successfully! Total transitions: {total_steps}", flush=True)
 
if __name__ == "__main__":
    # Ensure STABLEWM_HOME is set on D: to prevent C: disk space issues
    stable_wm_home = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))
    
    # Let's collect train & validation datasets
    train_path = os.path.join(stable_wm_home, "flappybird_expert_train.h5")
    val_path = os.path.join(stable_wm_home, "flappybird_expert_val.h5")
    
    collect_dataset(num_episodes=20, output_path=train_path)
    collect_dataset(num_episodes=5, output_path=val_path)
