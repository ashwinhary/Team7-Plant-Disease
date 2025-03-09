import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Enable Mixed Precision for CPU Speed-Up
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Set dataset paths
DATASET_PATH = "/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "validation")

# Ensure dataset paths exist
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Dataset directories not found: {TRAIN_DIR} or {VAL_DIR}")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load images in batches
BATCH_SIZE = 16  # Reduce batch size for CPU training
IMG_SIZE = (224, 224)  # MobileNetV2 standard input size
NUM_CLASSES = 39  # Adjust based on dataset

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Compute class weights to handle imbalance
labels = train_generator.classes  # Get labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Load MobileNetV2 (Pretrained)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Initially freeze base model layers
base_model.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)

# Define initial model (Frozen Layers)
model = Model(inputs=base_model.input, outputs=x)

# Compile model with frozen base
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks (Checkpoint + Early Stopping)
checkpoint = ModelCheckpoint("model_best.h5", save_best_only=True, monitor="val_accuracy", mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train initial model (Frozen layers)
model.fit(
    train_generator,
    epochs=10,  # Train first 10 epochs with frozen layers
    validation_data=val_generator,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,  # Apply class weights
    callbacks=[checkpoint, early_stop]
)

# Unfreeze last 20 layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compile model again with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Reduce LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue fine-tuning model
model.fit(
    train_generator,
    epochs=20,  # Fine-tune for additional 20 epochs
    validation_data=val_generator,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,  # Keep class weighting
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save("plant_disease_model_final.h5")

print("Training complete. Model saved.")