import os
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset(dataset_path):
    # Dictionary to hold the count of images per disease for each plant
    plant_diseases = {}
    total_images = 0

    # Walk through the dataset
    for plant_folder in os.listdir(dataset_path):
        plant_path = os.path.join(dataset_path, plant_folder)
        
        if os.path.isdir(plant_path):
            plant_diseases[plant_folder] = {}
            for disease_folder in os.listdir(plant_path):
                disease_path = os.path.join(plant_path, disease_folder)

                if os.path.isdir(disease_path):
                    image_count = len([f for f in os.listdir(disease_path) if f.endswith('.jpg') or f.endswith('.png')])
                    plant_diseases[plant_folder][disease_folder] = image_count
                    total_images += image_count
    
    return plant_diseases, total_images

def print_summary(plant_diseases, total_images):
    print(f"Total Images in the Dataset: {total_images}")
    print(f"\nPlant-wise Disease Count:")
    for plant, diseases in plant_diseases.items():
        print(f"\n{plant}:")
        for disease, count in diseases.items():
            print(f"   {disease}: {count} images")

def show_sample_images(dataset_path, plant_diseases, num_samples=5):
    print("\nDisplaying sample images from different classes...")
    
    sample_classes = random.sample(list(plant_diseases.keys()), min(num_samples, len(plant_diseases)))

    fig, axes = plt.subplots(1, len(sample_classes), figsize=(15, 5))
    
    for ax, plant in zip(axes, sample_classes):
        disease = random.choice(list(plant_diseases[plant].keys()))
        disease_path = os.path.join(dataset_path, plant, disease)

        # Check if the disease folder exists and contains images
        if not os.path.exists(disease_path) or len(os.listdir(disease_path)) == 0:
            print(f"Warning: Disease folder '{disease_path}' is empty or not found.")
            continue
        
        try:
            image_name = random.choice(os.listdir(disease_path))
            image_path = os.path.join(disease_path, image_name)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img)
            ax.set_title(f"{plant}\n{disease}")
            ax.axis("off")
        except Exception as e:
            print(f"Error loading image from {disease_path}: {e}")
            continue

    # Save the plot instead of showing it
    plt.tight_layout()
    plt.savefig("sample_images_from_classes.png")  # Save the image as a PNG file
    print("Sample images saved as 'sample_images_from_classes.png'")