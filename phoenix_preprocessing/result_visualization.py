import cv2
import numpy as np
from mediapipe.tasks.python import vision

def draw_landmarks(image, landmarks, connections, color=(0, 255, 0), radius=3, thickness=2):
    """
    Draws skeletal landmarks and their connections on a given image canvas.
    Handles zero-padded arrays by skipping the drawing process.
    """
    height, width, _ = image.shape
    points = []
    
    # Extract and scale coordinates
    for lm in landmarks:
        # Check if missing (zero-padding)
        if np.all(lm == 0):
            points.append(None)
        else:
            cx, cy = int(lm[0] * width), int(lm[1] * height)
            points.append((cx, cy))
            # Drawing
            cv2.circle(image, (cx, cy), radius, color, -1)
    
    # Connections between keypoints
    if connections:
        for connection in connections:
            idx1 = connection.start
            idx2 = connection.end
            
            if idx1 < len(points) and idx2 < len(points):
                p1 = points[idx1]
                p2 = points[idx2]
                if p1 is not None and p2 is not None:
                    cv2.line(image, p1, p2, color, thickness)

def visualize_sequence(npy_path, window_width=800, window_height=800):
    """
    Loads a .npy file containing the extracted sequence (1659 values per frame) 
    and visualizes the frames chronologically.
    """
    try:
        data = np.load(npy_path)
    except FileNotFoundError:
        print(f"Error: The file {npy_path} was not found.")
        return

    print(f"Visualizing sequence with shape: {data.shape}")
    print("Press 'Q' to stop the visualization.")

    # Retrieve connection maps
    pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    # TESSELATION for a complete facial mesh visualization
    face_connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

    for frame_data in data:
        # Blank black canvas
        image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # Reconstruct the feature vectors into (N, 3) coordinate matrices in correct orders
        pose_data = frame_data[0:99].reshape(33, 3)
        face_data = frame_data[99:1533].reshape(478, 3)
        lh_data = frame_data[1533:1596].reshape(21, 3)
        rh_data = frame_data[1596:1659].reshape(21, 3)
        
        # Render skeleton with distinct colors (white, green, red)
        draw_landmarks(image, pose_data, pose_connections, color=(255, 255, 255)) 
        draw_landmarks(image, lh_data, hand_connections, color=(0, 255, 0))       
        draw_landmarks(image, rh_data, hand_connections, color=(0, 0, 255))       
        
        # Face rendering with smaller points and lines (cyan)
        draw_landmarks(image, face_data, face_connections, color=(255, 255, 0), radius=1, thickness=1)
        
        cv2.imshow('Sign Language Skeleton Visualization', image)
        
        # Break the loop if Q is presed
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Direct path for samples
    sample_file_path = 'sample_results/01April_2010_Thursday_heute-6697.npy' 
    visualize_sequence(sample_file_path)