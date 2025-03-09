import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
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

# Load MobileNetV2 (Pretrained)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers (for transfer learning)
base_model.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)  # Ensure this layer is connected properly

# Define final model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks (Checkpoint + Early Stopping)
checkpoint = ModelCheckpoint("model_best.h5", save_best_only=True, monitor="val_accuracy", mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train model
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save("plant_disease_model_final.h5")

print("Training complete. Model saved.")