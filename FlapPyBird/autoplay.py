import os
import sys
import torch
import numpy as np
from pathlib import Path

# Thêm đường dẫn tới thư mục gốc
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "NCKH" / "le-wm"))
from src.flappy_env import FlappyEnv

def autoplay(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    stable_wm_home = os.environ.get("STABLEWM_HOME", "D:\\stable-wm")
    
    # 1. Load JEPA Model
    print("Loading JEPA Model...")
    spt_module = torch.load(ckpt_path, weights_only=False, map_location=device)
    jepa = spt_module.model.eval().to(device)
    
    # 2. Load State Predictor
    print("Loading State Predictor...")
    try:
        from train_state_predictor import StatePredictor
    except ImportError:
        print("Cannot import train_state_predictor. Make sure it is in PYTHONPATH.")
        sys.exit(1)
        
    state_predictor_path = os.path.join(stable_wm_home, "state_predictor.pt")
    if not os.path.exists(state_predictor_path):
        print(f"Error: State predictor not found at {state_predictor_path}")
        print("Please train it first using: python train_state_predictor.py <ckpt_path>")
        sys.exit(1)
        
    state_predictor = StatePredictor(input_dim=192, output_dim=3).to(device)
    state_predictor.load_state_dict(torch.load(state_predictor_path, map_location=device))
    state_predictor.eval()
    
    # 3. Setup Environment
    os.environ.pop("SDL_VIDEODRIVER", None) # Ensure GUI is shown
    env = FlappyEnv(mute=False)
    
    # 4. MPC Parameters
    H_SIZE = 3   # History size
    N_CANDIDATES = 500  # Number of random action sequences
    PLAN_HORIZON = 15   # Look ahead steps
    
    obs, info = env.reset()
    
    # Keep history of observations
    obs_history = [obs for _ in range(H_SIZE)]
    action_history = [0 for _ in range(H_SIZE)]
    
    done = False
    step = 0
    
    print("Autoplay starting...")
    while not done:
        pixels = np.array(obs_history, dtype=np.float32) / 255.0  # (H_SIZE, 224, 224, 3)
        pixels = torch.from_numpy(pixels).permute(0, 3, 1, 2)     # (H_SIZE, 3, 224, 224)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pixels = (pixels - mean) / std
        
        # Expand for candidates: (B, S, T_hist, C, H, W)
        pixels = pixels.unsqueeze(0).unsqueeze(0).expand(1, N_CANDIDATES, H_SIZE, 3, 224, 224).to(device)
        
        act_hist = torch.tensor(action_history, dtype=torch.float32).view(1, 1, H_SIZE, 1)
        act_hist = act_hist.expand(1, N_CANDIDATES, H_SIZE, 1).to(device)
        
        future_acts = torch.randint(0, 2, (1, N_CANDIDATES, PLAN_HORIZON, 1), dtype=torch.float32, device=device)
        actions = torch.cat([act_hist, future_acts], dim=2)
        
        info_dict = {"pixels": pixels}
        with torch.no_grad():
            info_dict = jepa.rollout(info_dict, actions, history_size=H_SIZE)
            pred_emb = info_dict["predicted_emb"] # (1, S, T_hist + PLAN_HORIZON + 1, D)
            pred_states = state_predictor(pred_emb) # (1, S, T_total, 3)
            
            pipe_dist_x = pred_states[0, :, :, 0]
            pipe_gap_center_y = pred_states[0, :, :, 1]
            bird_y = pred_states[0, :, :, 2]
            
            cost = torch.zeros(N_CANDIDATES, device=device)
            
            for t in range(H_SIZE, pred_states.size(2)):
                cost += torch.where(bird_y[:, t] > 400, 10000.0, 0.0) # Hit floor
                cost += torch.where(bird_y[:, t] < 0, 10000.0, 0.0)   # Hit ceiling
                
                # If crossing pipe, strongly penalize vertical distance to gap center
                crossing = (pipe_dist_x[:, t] > -40) & (pipe_dist_x[:, t] < 60)
                gap_distance = torch.abs(bird_y[:, t] - pipe_gap_center_y[:, t])
                
                cost += torch.where(crossing, gap_distance * 100.0, gap_distance * 1.0)
                
                # Small penalty for flapping to avoid jittering
                if t < H_SIZE + PLAN_HORIZON:
                    cost += actions[0, :, t, 0] * 10.0
        
        best_idx = torch.argmin(cost).item()
        best_action = future_acts[0, best_idx, 0, 0].item()
        
        obs, reward, done, info = env.step(int(best_action))
        
        obs_history.pop(0)
        obs_history.append(obs)
        action_history.pop(0)
        action_history.append(best_action)
        
        step += 1
        if step % 100 == 0:
            print(f"Step {step}, Score: {info['score']}")
            
    print(f"Game Over! Final Score: {info['score']}")
    env.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autoplay.py <path_to_jepa_checkpoint.ckpt>")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    autoplay(ckpt_path)
