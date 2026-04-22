import cv2
import os
from mtcnn import MTCNN

detector = MTCNN()

input_root = "dataset"
output_root = "dataset_cropped"

os.makedirs(output_root, exist_ok=True)

# Loop over all persons
for person in os.listdir(input_root):
    input_dir = os.path.join(input_root, person)
    output_dir = os.path.join(output_root, person)

    if not os.path.isdir(input_dir):
        continue

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    print(f"Processing: {person}")

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            x, y = abs(x), abs(y)

            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)

            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))

            save_path = os.path.join(output_dir, f"{count}.jpg")
            cv2.imwrite(save_path, face)

            count += 1

    print(f"Done cropping for {person}")

print("✅ All faces cropped successfully!")