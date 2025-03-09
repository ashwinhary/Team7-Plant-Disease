import os
import shutil
import random

# Set the base path of your dataset
dataset_dir = "/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/"

# List all subdirectories (class names)
classes = os.listdir(dataset_dir)

# Create the train, validation, and test directories
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split each class into train, validation, and test
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)

    # Check if it's a directory
    if os.path.isdir(class_dir):
        # Create the subdirectories for train, validation, and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # List all images in the class directory
        images = os.listdir(class_dir)

        # Shuffle the list of images
        random.shuffle(images)

        # Calculate the number of images for each split (80% train, 10% validation, 10% test)
        train_size = int(len(images) * 0.8)
        val_size = int(len(images) * 0.1)
        test_size = len(images) - train_size - val_size

        # Split the images into train, validation, and test
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # Move the images into the corresponding directories
        for image in train_images:
            shutil.move(os.path.join(class_dir, image), os.path.join(train_dir, class_name, image))

        for image in val_images:
            shutil.move(os.path.join(class_dir, image), os.path.join(val_dir, class_name, image))

        for image in test_images:
            shutil.move(os.path.join(class_dir, image), os.path.join(test_dir, class_name, image))

print("Dataset split completed!")