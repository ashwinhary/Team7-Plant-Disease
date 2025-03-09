import tensorflow as tf
import numpy as np

# Step 1: Convert the Keras model to TensorFlow Lite
keras_model_path = "plant_disease_model_final.keras"
tflite_model_path = "plant_disease_model.tflite"

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_model_path}")

# Step 2: Load TFLite model and run inference
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare a test image (dummy example, replace with actual test image)
input_shape = input_details[0]['shape']
test_image = np.random.rand(*input_shape).astype(np.float32)  # Replace with real image preprocessing

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], test_image)

# Run inference
interpreter.invoke()

# Get the output
output = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output)

print(f"Predicted Class: {predicted_class}")