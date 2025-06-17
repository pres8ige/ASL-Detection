import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model(r'C:\Users\Kaiwalya\OneDrive\Desktop\asl detection\mobilenet_asl_best.h5')

# Set image size same as model input
image_size = 224

# Class labels (26 classes A-Z)
classes = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            h, w, _ = image.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Crop hand region
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            hand_img = image[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess for model
            hand_img = cv2.resize(hand_img, (image_size, image_size))
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)  # Shape: (1, 64, 64, 3)

            # Predict
            predictions = model.predict(hand_img)
            class_id = int(np.argmax(predictions))
            confidence = float(predictions[0][class_id])

            # Safety check
            if class_id < len(classes):
                label = f"{classes[class_id]} ({confidence * 100:.2f}%)"
            else:
                label = f"Unknown ({confidence * 100:.2f}%)"

            # Display result
            cv2.putText(image, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                          (0, 255, 0), 2)

    cv2.imshow("ASL Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
