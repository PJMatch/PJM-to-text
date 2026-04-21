import argparse
import numpy as np
from PIL import Image, ImageDraw

# connection definitions
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

def visualize_npy_to_gif_pil(npy_path, output_gif="visualization_fast.gif", fps=20, img_size=600):
    print(f"Loading data from: {npy_path}...")
    data = np.load(npy_path, allow_pickle=True)
    frames = []

    print(f"Loaded {len(data)} frames. Drawing skeleton...")

    # scaling points
    def get_px_coords(points_list):
        if not points_list: return []
        pts = np.array(points_list)
        if pts.ndim != 2: return []
        return (pts[:, :2] * img_size).astype(int)

    # drwaing process
    for frame_idx, frame_data in enumerate(data):
        img = Image.new('RGB', (img_size, img_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        pose_px = get_px_coords(frame_data.get('pose', []))
        face_px = get_px_coords(frame_data.get('face', []))
        lh_px = get_px_coords(frame_data.get('lh', []))
        rh_px = get_px_coords(frame_data.get('rh', []))

        def draw_lines(coords, connections, color, width=2):
            if len(coords) == 0: return
            for u, v in connections:
                if u < len(coords) and v < len(coords):
                    p1 = tuple(coords[u])
                    p2 = tuple(coords[v])
                    draw.line([p1, p2], fill=color, width=width)

        draw_lines(pose_px, POSE_CONNECTIONS, color=(0, 0, 255), width=3)   
        draw_lines(lh_px, HAND_CONNECTIONS, color=(255, 0, 0), width=2)     
        draw_lines(rh_px, HAND_CONNECTIONS, color=(0, 128, 0), width=2)     

        def draw_dots(coords, color, radius=3):
            for x, y in coords:
                draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=color)

        draw_dots(face_px, color=(200, 200, 200), radius=1)  
        draw_dots(pose_px, color=(0, 0, 255), radius=4)
        draw_dots(lh_px, color=(255, 0, 0), radius=3)
        draw_dots(rh_px, color=(0, 128, 0), radius=3)

        frames.append(img)
        
        # progress bar
        if frame_idx % 20 == 0:
            print(f"Rendered frame {frame_idx}/{len(data)}...")

    # save to gif
    print("Saving animation to file...")
    duration_per_frame = int(1000 / fps)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_frame,
        loop=0 # 0 = loop
    )
    print(f"Success! Animation saved as {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skeleton visualization")
    parser.add_argument("--input", type=str, required=True, help="Path to the .npy file")
    parser.add_argument("--output", type=str, default="visualization.gif", help="Output GIF filename")
    parser.add_argument("--fps", type=int, default=20, help="FPS")
    parser.add_argument("--size", type=int, default=600, help="Resolution")
    
    args = parser.parse_args()
    visualize_npy_to_gif_pil(args.input, args.output, args.fps, args.size)