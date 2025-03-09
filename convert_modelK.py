import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# ? Load the trained model
h5_model_path = "plant_disease_model.h5"
model = load_model(h5_model_path)

# ? Print model summary to verify the expected input shape
model.summary()

# ? Get expected input shape from the model
expected_shape = model.input_shape  # Example: (None, 148, 148, 3)
height, width = expected_shape[1], expected_shape[2]

# ? Load and preprocess a sample image for prediction
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, target_size)  # Resize to match model's expected input
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ? Test with a sample image
image_path = "/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/test/Tomato___Leaf_Mold/image (54).JPG"  # Replace with an act image
input_image = preprocess_image(image_path, (height, width))

# ? Make prediction
predictions = model.predict(input_image)
print("Predictions:", predictions)

# ? Get the predicted class
predicted_class = np.argmax(predictions, axis=1)[0]
print("Predicted Class:", predicted_class)
