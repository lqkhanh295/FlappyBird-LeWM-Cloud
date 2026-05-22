import h5py
import numpy as np

def patch_h5(filepath):
    print(f"Patching {filepath}")
    with h5py.File(filepath, 'a') as f:
        ep_idx = f['episode_idx'][:]
        
        changes = np.where(np.diff(ep_idx) != 0)[0] + 1
        offsets = np.insert(changes, 0, 0)
        lengths = np.append(np.diff(offsets), len(ep_idx) - offsets[-1])
        
        offsets = offsets.astype(np.int32)
        lengths = lengths.astype(np.int32)
        
        if 'ep_len' in f:
            del f['ep_len']
        if 'ep_offset' in f:
            del f['ep_offset']
            
        f.create_dataset('ep_len', data=lengths, compression="gzip")
        f.create_dataset('ep_offset', data=offsets, compression="gzip")
        
        print(f"Added {len(lengths)} episodes. Total steps: {np.sum(lengths)}")

patch_h5("D:/stable-wm/flappybird_expert_train.h5")
patch_h5("D:/stable-wm/flappybird_expert_val.h5")
