import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    rotation_range=20,       # Rotation
    width_shift_range=0.2,   # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,         # Shearing
    zoom_range=0.2,          # Zooming
    horizontal_flip=True,    # Horizontal flipping
    fill_mode='nearest')


# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
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
IMAGES_DIR = os.path.join('.', 'cropped_image_augmented')

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

number_of_augmented_images = 5  # Define the number of augmented images

augmented_images = [datagen.random_transform(image.img_to_array(cv2.resize(cv2.cvtColor(cv2.imread(other_image_path), cv2.COLOR_BGR2RGB), (224, 224)))) for _ in range(number_of_augmented_images)]
augmented_images_preprocessed = [preprocess_input(np.expand_dims(img, axis=0)) for img in augmented_images]

# Extract features for each augmented image of the other image
other_features_list = [extract_features(img, resnet_model) for img in augmented_images_preprocessed]

# ... [previous code] ...

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_img_path = os.path.join(IMAGES_DIR, f'cropped_{box_count}.jpg')
        cv2.imwrite(cropped_img_path, cropped_img)

        # Generate augmented images for the cropped image
        augmented_cropped_images = [datagen.random_transform(image.img_to_array(cv2.resize(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), (224, 224)))) for _ in range(number_of_augmented_images)]
        augmented_cropped_images_preprocessed = [preprocess_input(np.expand_dims(img, axis=0)) for img in augmented_cropped_images]

        # Initialize maximum similarity score
        max_similarity_score = 0

        # Extract features for each augmented cropped image
        for aug_cropped_img in augmented_cropped_images_preprocessed:
            cropped_features = extract_features(aug_cropped_img, resnet_model)

            # Compare with each augmented image of the other image
            for other_features in other_features_list:
                similarity_score = calculate_similarity(cropped_features, other_features)
                max_similarity_score = max(max_similarity_score, similarity_score)

        # Print the highest similarity score for this cropped image
        print(f"Highest similarity score for cropped_{box_count}.jpg: {max_similarity_score:.4f}")

        box_count += 1
