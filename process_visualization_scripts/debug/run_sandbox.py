import numpy as np
import torch
import os

from model_debug import CoSign1SModel

def simulate_dataloader_pipeline(npy_path):
    print(f"Loading raw dict data from {npy_path}...")
    raw_data = np.load(npy_path, allow_pickle=True)
    T = len(raw_data)
    V = 553
    C = 3
    
    tensor_data = np.zeros((T, V, C), dtype=np.float32)
    
    for t, frame in enumerate(raw_data):
        pose = np.array(frame.get('pose', np.zeros((33, C))))
        face = np.array(frame.get('face', np.zeros((478, C))))
        lh = np.array(frame.get('lh', np.zeros((21, C))))
        rh = np.array(frame.get('rh', np.zeros((21, C))))
        
        def safe_assign(dest, src, start_idx, end_idx):
            v_len = end_idx - start_idx
            if src.ndim == 2 and len(src) == v_len:
                c_len = min(C, src.shape[1])
                dest[t, start_idx:end_idx, :c_len] = src[:, :c_len]

        safe_assign(tensor_data, pose, 0, 33)
        safe_assign(tensor_data, face, 33, 511)
        safe_assign(tensor_data, lh, 511, 532)
        safe_assign(tensor_data, rh, 532, 553)
        
    frames_btvc = torch.tensor(tensor_data).unsqueeze(0)
    print(f"Dataloader output shape: {frames_btvc.shape} [B, T, V, C]")
    
    frames_bctv = frames_btvc.permute(0, 3, 1, 2)
    print(f"Input to model shape:    {frames_bctv.shape} [B, C, T, V]")
    
    return frames_bctv, T

if __name__ == "__main__":
    file_path = "9_97_20260324_194957.npy"
    
    x, time_frames = simulate_dataloader_pipeline(file_path)
    lengths = torch.tensor([time_frames])
    
    print("Initializing CoSign1SModel sandbox...")
    model = CoSign1SModel(num_classes=1000)
    model.eval()
    
    print("Pushing data through the network...")
    with torch.no_grad():
        _ = model(x, lengths)
        
    print("Finished! Check your folder for the new checkpoint .npy files.")