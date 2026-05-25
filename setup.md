pip install opencv-python numpy mtcnn tensorflow keras-facenet


Step 1: Run python capture.py to capture face images and store them in the dataset folder.
Step 2: Run python detect_crop.py to detect faces and save cropped images.
Step 3: Run python embeddings.py to convert faces into numerical embeddings.
Step 4: Run python recognize.py to start the webcam and perform real-time recognition.
Step 5: Pipeline → Capture → Detect & Crop → Embeddings → Recognize.
