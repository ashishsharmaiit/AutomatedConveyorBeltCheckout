from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


# Load the ResNet50 model pre-trained on ImageNet data
model = ResNet50(weights='imagenet', include_top=False)

img_path = 'frame_48.jpg'


def load_and_preprocess_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Expand dimensions to fit model input format
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    img_array = preprocess_input(img_array)

    return img_array

def extract_features(img_array, model):
    # Get feature vector
    features = model.predict(img_array)

    # Flatten the features to one dimension
    features_flatten = features.flatten()

    return features_flatten

# Example image path

# Load and preprocess the image
img_array = load_and_preprocess_image(img_path)

# Extract features
features = extract_features(img_array, model)

# Features now contains the high-dimensional vector for your image

# Extract features
features = extract_features(img_array, model)

feature_list = features.tolist()
print("Full Feature Vector:", feature_list)
