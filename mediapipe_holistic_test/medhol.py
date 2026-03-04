import cv2
import mediapipe as mp
import time

# Init
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_skeleton(frame, results):
    """Nakłada siatkę i punkty na wykryte części ciała."""
    # Rysowanie twarzy
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
        )
        
    # Rysowanie postawy - ramiona tułuw
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
        
    # Rysowanie dłoni 
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Konfiguracja kamery
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_time = 0

# Uruchomienie modelu - complexity=0 -> za słabe, complexity=1 w miare ok, complexity=2 za trudne dla mojego cpu
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as holistic:
    print("q to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error kamery")
            break
        
        # MediaPipe wymaga formatu RGB, a OpenCV domyślnie używa BGR
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analiza klatki
        results = holistic.process(rgb_frame)
        
        frame.flags.writeable = True
        
        #Rysowanie
        draw_skeleton(frame, results)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f'FPS: {int(fps)}', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("System PJM", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()