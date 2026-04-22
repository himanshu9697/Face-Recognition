import cv2
import os

# Enter user name:
name = input("Enter your name: ")

# Folder will be created for that person:
dataset_path = "dataset/" + name
os.makedirs(dataset_path, exist_ok=True)

# Webcam will open:
cap = cv2.VideoCapture(0)

count = 0

print("Press 'c' to capture image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show frame
    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1)

    # Press 'c' to capture
    if key == ord('c'):
        img_path = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    # Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()