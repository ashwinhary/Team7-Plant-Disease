# compile_model.py

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.saving import save_model
from custom_loss import weighted_loss  # Import the custom loss function

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss=weighted_loss, metrics=['accuracy'])
    
    return model

# Define input shape and number of classes
input_shape = (64, 64, 3)  # Modify according to your dataset
num_classes = 39  # Modify according to your number of classes

# Build and train the model (using your dataset here)
model = build_model(input_shape, num_classes)

# Train the model with your data
# model.fit(...)

# Save the model in the .keras format
model.save('/home/exouser/plant_disease_model.keras')
print("Model saved successfully.")