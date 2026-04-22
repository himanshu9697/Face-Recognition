import cv2
import numpy as np
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

# Load embeddings
known_embeddings = []
known_names = []

for file in os.listdir("embeddings"):
    name = file.split(".")[0]
    data = np.load(os.path.join("embeddings", file))

    for emb in data:
        known_embeddings.append(emb)
        known_names.append(name)

known_embeddings = np.array(known_embeddings)

# Start webcam
cap = cv2.VideoCapture(0)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

threshold = 0.4  # adjust later if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)

        # Add margin (same as before)
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face_img = frame[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)

        embedding = embedder.embeddings(face_img)[0]

        # Compare with known embeddings
        distances = [cosine_distance(embedding, e) for e in known_embeddings]
        min_dist = min(distances)
        index = distances.index(min_dist)

        if min_dist < threshold:
            name = known_names[index]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Draw box + name
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        confidence = (1 - min_dist) * 100
        cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()