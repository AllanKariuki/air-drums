# import cv2
# import mediapipe as mp
# import numpy as np

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.9,
#     min_tracking_confidence=0.9
# )

# mp_draw = mp.solutions.drawing_utils

# def is_thumb_up(hand_landmarks, frame_shape):
#     h, w, _ = frame_shape

#     # Extract landmark positions
#     thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
#     thumb_ip = hand_landmarks.landmark[3]   # Thumb IP joint
#     thumb_mcp = hand_landmarks.landmark[2]  # Thumb MCP joint
#     wrist = hand_landmarks.landmark[0]      # Wrist
    
#     # Get finger tips and bases for other fingers
#     index_tip = hand_landmarks.landmark[8]
#     middle_tip = hand_landmarks.landmark[12]
#     ring_tip = hand_landmarks.landmark[16]
#     pinky_tip = hand_landmarks.landmark[20]
    
#     index_pip = hand_landmarks.landmark[6]
#     middle_pip = hand_landmarks.landmark[10]
#     ring_pip = hand_landmarks.landmark[14]
#     pinky_pip = hand_landmarks.landmark[18]
    
#     # Convert to pixel coordinates
#     thumb_tip_y = thumb_tip.y * h
#     thumb_mcp_y = thumb_mcp.y * h
#     wrist_y = wrist.y * h
    
#     # Check if thumb is pointing up (relative to wrist)
#     thumb_up = thumb_tip_y < thumb_mcp_y < wrist_y
    
#     # Check if other fingers are curled (fingertip below PIP joint)
#     index_curled = index_tip.y > index_pip.y
#     middle_curled = middle_tip.y > middle_pip.y
#     ring_curled = ring_tip.y > ring_pip.y
#     pinky_curled = pinky_tip.y > pinky_pip.y
    
#     # Check that the thumb is extended significantly
#     thumb_distance = np.linalg.norm(
#         np.array([thumb_tip.x * w, thumb_tip.y * h]) - 
#         np.array([wrist.x * w, wrist.y * h])
#     )
#     thumb_extended = thumb_distance > 0.15 * h
    
#     # Determine handedness (left or right) to adjust checks
#     is_right_hand = thumb_tip.x < index_tip.x  # Simple heuristic
    
#     # A thumbs up is when:
#     # 1. Thumb is pointing up
#     # 2. Thumb is extended
#     # 3. Other fingers are curled
#     return (thumb_up and 
#             thumb_extended and 
#             index_curled and 
#             middle_curled and 
#             ring_curled and 
#             pinky_curled)

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for a mirrored view
#     frame = cv2.flip(frame, 1)

#     # Convert BGR to RGB (mediapipe uses RGB input)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect hands
#     results = hands.process(frame_rgb)

#     # Flag to track if we've found a thumbs up
#     thumb_up_detected = False
    
#     # Check if any hands are detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw hand landmarks
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Check if the thumb is up
#             if is_thumb_up(hand_landmarks, frame.shape):
#                 thumb_up_detected = True
    
#     # Only display the thumbs up text if we detected the gesture
#     if thumb_up_detected:
#         cv2.putText(frame, "👍 Thumbs Up!", (50, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#     cv2.imshow("Gesture Recognition", frame)

#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

def is_thumb_up(hand_landmarks, frame_shape):
    h, w, _ = frame_shape

    # Extract landmark positions
    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
    thumb_ip = hand_landmarks.landmark[3]   # Thumb IP joint
    thumb_mcp = hand_landmarks.landmark[2]  # Thumb MCP joint
    wrist = hand_landmarks.landmark[0]      # Wrist
    
    # Get finger tips and bases for other fingers
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    index_pip = hand_landmarks.landmark[6]
    middle_pip = hand_landmarks.landmark[10]
    ring_pip = hand_landmarks.landmark[14]
    pinky_pip = hand_landmarks.landmark[18]
    
    # Convert to pixel coordinates
    thumb_tip_y = thumb_tip.y * h
    thumb_mcp_y = thumb_mcp.y * h
    wrist_y = wrist.y * h
    
    # Check if thumb is pointing up (relative to wrist)
    thumb_up = thumb_tip_y < thumb_mcp_y < wrist_y
    
    # Check if other fingers are curled (fingertip below PIP joint)
    index_curled = index_tip.y > index_pip.y
    middle_curled = middle_tip.y > middle_pip.y
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y
    
    # Check that the thumb is extended significantly
    thumb_distance = np.linalg.norm(
        np.array([thumb_tip.x * w, thumb_tip.y * h]) - 
        np.array([wrist.x * w, wrist.y * h])
    )
    thumb_extended = thumb_distance > 0.15 * h
    
    # A thumbs up is when:
    # 1. Thumb is pointing up
    # 2. Thumb is extended
    # 3. Other fingers are curled
    return (thumb_up and 
            thumb_extended and 
            index_curled and 
            middle_curled and 
            ring_curled and 
            pinky_curled)

# Create a history buffer to smooth detection results
history_size = 10  # Number of frames to consider
detection_history = deque(maxlen=history_size)
detection_threshold = 0.7  # Percentage of frames needed to consider as detected

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (mediapipe uses RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    # Reset detection for current frame
    current_frame_detection = False
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the thumb is up
            if is_thumb_up(hand_landmarks, frame.shape):
                current_frame_detection = True
    
    # Add current frame's detection result to history
    detection_history.append(current_frame_detection)
    
    # Calculate the percentage of frames with thumbs up detection
    detection_ratio = sum(detection_history) / len(detection_history)
    
    # Only display the thumbs up text if the detection ratio is above threshold
    if detection_ratio >= detection_threshold:
        cv2.putText(frame, "👍 Thumbs Up!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # Optionally display the detection confidence (for debugging)
    cv2.putText(frame, f"Confidence: {detection_ratio:.2f}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()