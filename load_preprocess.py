import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for the directories
train_dir = '/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/train'
validation_dir = '/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/validation'
test_dir = '/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/test'

# Data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load images and apply augmentation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)