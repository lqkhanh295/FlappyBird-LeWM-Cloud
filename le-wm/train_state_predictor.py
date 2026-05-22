import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class StatePredictor(nn.Module):
    def __init__(self, input_dim=192, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    stable_wm_home = os.environ.get("STABLEWM_HOME", "D:\\stable-wm")
    
    if len(sys.argv) < 2:
        print("Usage: python train_state_predictor.py <path_to_jepa_checkpoint.ckpt>")
        return
        
    ckpt_path = sys.argv[1]
    
    print("Loading JEPA model...")
    spt_module = torch.load(ckpt_path, weights_only=False, map_location=device)
    jepa = spt_module.model.eval().to(device)
    
    print("Loading dataset...")
    train_h5 = os.path.join(stable_wm_home, "flappybird_expert_train.h5")
    with h5py.File(train_h5, 'r') as f:
        # Load up to 2000 samples for fast training
        n_samples = min(2000, f['pixels'].shape[0])
        pixels = f['pixels'][:n_samples]
        states = f['state'][:n_samples]
        
    print(f"Loaded {n_samples} transitions. Extracting latent embeddings...")
    
    # Process images: to float32, normalize (using ImageNet stats as in lewm), to (B, C, H, W)
    pixels = torch.from_numpy(pixels).float()
    pixels = pixels.permute(0, 3, 1, 2) # (N, C, H, W)
    pixels = pixels / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels = (pixels - mean) / std
    
    batch_size = 32
    emb_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting"):
            batch_pixels = pixels[i:i+batch_size].to(device)
            # Add time dimension: (B, T, C, H, W) where T=1
            batch_pixels = batch_pixels.unsqueeze(1)
            info = {"pixels": batch_pixels}
            info = jepa.encode(info)
            emb = info["emb"].squeeze(1).cpu() # (B, D)
            emb_list.append(emb)
            
    embeddings = torch.cat(emb_list, dim=0) # (N, D)
    targets = torch.from_numpy(states)      # (N, 3)
    
    dataset = TensorDataset(embeddings, targets)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("Training State Predictor...")
    predictor = StatePredictor(input_dim=embeddings.shape[-1], output_dim=3).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(20):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = predictor(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/20 - Loss: {total_loss/len(loader):.4f}")
        
    save_path = os.path.join(stable_wm_home, "state_predictor.pt")
    torch.save(predictor.state_dict(), save_path)
    print(f"State Predictor saved to {save_path}!")

if __name__ == "__main__":
    train()
