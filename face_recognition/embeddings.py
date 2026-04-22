import os
import numpy as np
import cv2
from keras_facenet import FaceNet

embedder = FaceNet()

input_dir = "dataset_cropped"
output_dir = "embeddings"

os.makedirs(output_dir, exist_ok=True)

# Loop over each person
for person in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person)

    # Skip if not a folder
    if not os.path.isdir(person_path):
        continue

    print(f"Processing: {person}")

    embeddings = [] # vector initialisation

    # Loop over images
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize for FaceNet
        img = cv2.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)

        embedding = embedder.embeddings(img)[0]
        embeddings.append(embedding)

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Save embeddings
    save_path = os.path.join(output_dir, f"{person}.npy")
    np.save(save_path, embeddings)

    print(f"Saved embeddings for {person}")

print("✅ All embeddings generated successfully!")