import numpy as np
import torch

def export_tensor_for_pil(tensor, filename="checkpoint.npy"):
    """
    Converts the [B, C, T, V] tensor back to a dictionary format 
    that our PIL visualization script can understand.
    """
    # Move video from batch to CPU
    t_data = tensor[0].detach().cpu().numpy()
    
    # Swap axes: [C, T, V] -> [T, V, C]
    t_data = np.transpose(t_data, (1, 2, 0))
    
    T = t_data.shape[0]
    frames = []
    
    for t in range(T):
        frame_dict = {
            'pose': t_data[t, 0:33, :].tolist(),
            'face': t_data[t, 33:511, :].tolist(),
            'lh': t_data[t, 511:532, :].tolist(),
            'rh': t_data[t, 532:553, :].tolist()
        }
        frames.append(frame_dict)
    
    np.save(filename, frames)
    print(f"[DEBUG] Saved skeleton to: {filename}")

def export_latent_heatmap(tensor, filename="checkpoint_3_latent.npy"):
    """
    Dumps the [1024, T] feature matrix from the STGCN layer.
    """
    # Extract features for the first batch item
    latent = tensor[0].detach().cpu().numpy()
    np.save(filename, latent)
    print(f"[DEBUG] Saved STGCN features to: {filename}")