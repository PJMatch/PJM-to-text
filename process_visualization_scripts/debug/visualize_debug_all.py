import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# --- CONNECTION DEFINITIONS ---
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), 
    (23, 25), (24, 26), (25, 27), (26, 28), 
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10) 
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       
    (0, 5), (5, 6), (6, 7), (7, 8),       
    (5, 9), (9, 10), (10, 11), (11, 12),  
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)                                
]

# NEW: Face contour connections for clear visualization
FACE_CONNECTIONS = [
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), 
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), 
    (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), 
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), 
    (54, 103), (103, 67), (67, 109), (109, 10),
    # Lips outer
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), 
    (321, 375), (375, 291), (291, 409), (409, 270), (270, 269), (269, 267), (267, 0), (0, 37), 
    (37, 39), (39, 40), (40, 61),
    # Left Eye
    (33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 33),
    # Right Eye
    (362, 385), (385, 387), (387, 263), (263, 373), (373, 380), (380, 362)
]

def draw_skeleton(ax, frame_data, title, is_anchored=False):
    ax.clear()
    ax.set_title(title)
    
    # Set limits based on whether data is anchored or raw
    if is_anchored:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.5, -0.5)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
    
    ax.set_aspect('equal')

    def plot_group(coords, connections, color, point_size=2):
        if not coords: return
        pts = np.array(coords)
        # Draw lines
        for u, v in connections:
            if u < len(pts) and v < len(pts):
                ax.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], color=color, linewidth=1)
        # Draw dots
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=point_size)

    # Plot groups with different colors
    plot_group(frame_data.get('pose', []), POSE_CONNECTIONS, 'blue')
    plot_group(frame_data.get('face', []), FACE_CONNECTIONS, 'cyan', point_size=0.5)
    plot_group(frame_data.get('lh', []), HAND_CONNECTIONS, 'red')
    plot_group(frame_data.get('rh', []), HAND_CONNECTIONS, 'green')

def visualize_checkpoints():
    print("Loading checkpoints...")
    raw_skeletons = np.load("checkpoint_1_raw.npy", allow_pickle=True)
    anchored_skeletons = np.load("checkpoint_2_anchored.npy", allow_pickle=True)
    latent_features = np.load("checkpoint_3_latent.npy", allow_pickle=True) # [1024, T]

    num_frames = min(len(raw_skeletons), latent_features.shape[1])

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    heatmap = ax3.imshow(latent_features, aspect='auto', cmap='viridis', interpolation='nearest')
    ax3.set_title("Checkpoint 3: Latent Features (1024 dim)")
    ax3.set_xlabel("Time (Frames)")
    ax3.set_ylabel("Feature Index")
    plt.colorbar(heatmap, ax=ax3)
    
    time_line = ax3.axvline(x=0, color='red', linestyle='--')

    def update(t):
        draw_skeleton(ax1, raw_skeletons[t], "CP1: Raw Input")
        draw_skeleton(ax2, anchored_skeletons[t], "CP2: Anchored (Relative to Root)", is_anchored=True)
        
        time_line.set_xdata([t, t])
        
        if t % 20 == 0:
            print(f"Animating frame {t}/{num_frames}...")
        
        return ax1, ax2, time_line

    print("Generating animation...")
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    output_file = "full_debug_visualization.gif"
    print(f"Saving to {output_file} (this might take a minute)...")
    ani.save(output_file, writer='pillow', fps=20)
    
    plt.close()
    print(f"Done! Visualization saved as {output_file}")

if __name__ == "__main__":
    visualize_checkpoints()