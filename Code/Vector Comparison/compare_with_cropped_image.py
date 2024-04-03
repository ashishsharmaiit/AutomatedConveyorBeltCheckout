import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine

# Function to load and preprocess the image
def load_and_preprocess_image(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features
def extract_features(img_array, model):
    features = model.predict(img_array)
    features_flatten = features.flatten()
    return features_flatten

# Function to calculate cosine similarity
def calculate_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

# Load the ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False)

# YOLO Model and Cropping Logic
IMAGES_DIR = os.path.join('.', 'cropped_images')

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

frame_path = os.path.join('.', 'test_frames', 'frame_130.jpg')
frame = cv2.imread(frame_path)

model_path = os.path.join('.', 'BeltObjectModel', 'weights', 'best.pt')
model = YOLO(model_path)
threshold = 0.7

results = model(frame)[0]
box_count = 1


# Now compare this with another image
other_image_path = 'shaan.jpg'  # Replace with your other image path
other_image = cv2.imread(other_image_path)
other_image = cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB)
other_image = cv2.resize(other_image, (224, 224))
other_img_preprocessed = load_and_preprocess_image(other_image)
other_features = extract_features(other_img_preprocessed, resnet_model)


for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_img_filename = os.path.join(IMAGES_DIR, f'cropped_{box_count}.jpg')
        cv2.imwrite(cropped_img_filename, cropped_img)

        # Load and preprocess cropped image
        cropped_img_array = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_img_array = cv2.resize(cropped_img_array, (224, 224))
        cropped_img_preprocessed = load_and_preprocess_image(cropped_img_array)

        # Extract features
        cropped_features = extract_features(cropped_img_preprocessed, resnet_model)

        # Calculate similarity
        similarity_score = calculate_similarity(cropped_features, other_features)

        print(f"Similarity score between cropped_{box_count}.jpg and the other image: {similarity_score:.4f}")
        box_count += 1
