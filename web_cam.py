import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Display the frame
    cv2.imshow("Webcam feed", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release() # Release the webcam
cv2.destroyAllWindows() # Close the window