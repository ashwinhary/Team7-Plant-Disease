import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import numpy as np

# Build Hierarchical Model with Deep Supervision and Adaptive Loss Balancing
def build_model(num_classes_plant, num_classes_disease):
    # Load ResNet50 model with pre-trained weights (ImageNet)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Hierarchical Classification: First classify plant species, then the disease in that species
    plant_input = layers.Input(shape=(150, 150, 3))
    x = base_model(plant_input)
    x = layers.GlobalAveragePooling2D()(x)
    plant_output = layers.Dense(num_classes_plant, activation='softmax', name='plant_output')(x)

    # Disease Classification Layer (Deep Supervision)
    disease_input = layers.Input(shape=(150, 150, 3))
    y = base_model(disease_input)
    y = layers.GlobalAveragePooling2D()(y)
    disease_output = layers.Dense(num_classes_disease, activation='softmax', name='disease_output')(y)

    # Adding deep supervision to earlier layers of the network
    model = models.Model(inputs=[plant_input, disease_input], outputs=[plant_output, disease_output])

    # Compile the model with adaptive loss balancing
    model.compile(optimizer='adam', 
                  loss={'plant_output': 'categorical_crossentropy', 'disease_output': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    return model