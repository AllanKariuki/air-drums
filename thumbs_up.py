import cv2
import mediapipe as mp
import numpy as np

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

    # extract landmark positions
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    wrist = hand_landmarks.landmark[0]

    # Convert  to pixel cordinates
    thumb_tip_y = thumb_tip.y * h
    thumb_base_y = thumb_base.y * h
    wrist_y = wrist.y * h

    thumb_up = thumb_tip_y < thumb_base_y < wrist_y

    # Ensure that the thumb is far enough from the wrist to avoid false positives
    thumb_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([wrist.x, wrist.y]))

    return thumb_up and thumb_distance > 0.15

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (mediapipe uses RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect hands
    results = hands.process(frame_rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Check if the thumb is up
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # check if the thumb is up
            if is_thumb_up(hand_landmarks, frame.shape):
                cv2.putText(frame, "👍 Thumbs Up!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # thumb_tip = hand_landmarks.landmark[4]
            # thumb_base = hand_landmarks.landmark[2]

            # h, w, _ = frame.shape
            # thumb_tip_y = int(thumb_tip.y * h)
            # thumb_base_y = int(thumb_base.y * h)

            # # Check if the thumb is above the base (If thumb is up)
            # if thumb_tip_y < thumb_base_y:
            #     cv2.putText(frame, "Thumb is up", (50, 50), 
            # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv2.imshow("Gesture recognition", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()