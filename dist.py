import os
import matplotlib.pyplot as plt

def analyze_plant_images(dataset_path):
    plant_images = {}

    # Iterate through plant folders
    for plant_folder in os.listdir(dataset_path):
        plant_path = os.path.join(dataset_path, plant_folder)
        if os.path.isdir(plant_path):
            print(f"Found plant folder: {plant_folder}")  # Debugging line
            total_images = 0

            # Iterate through disease folders inside each plant folder
            for disease_folder in os.listdir(plant_path):
                disease_path = os.path.join(plant_path, disease_folder)
                if os.path.isdir(disease_path):
                    # Count the number of images for each disease folder
                    disease_count = len(os.listdir(disease_path))
                    total_images += disease_count

            if total_images > 0:
                plant_images[plant_folder] = total_images
                print(f"{plant_folder} has {total_images} images.")  # Debugging line

    return plant_images

def plot_plant_image_distribution(plant_images):
    # Sorting plant images by plant name for better presentation
    sorted_plants = sorted(plant_images.items(), key=lambda x: x[1], reverse=True)

    plant_names = [item[0] for item in sorted_plants]
    image_counts = [item[1] for item in sorted_plants]

    # Plot the data
    plt.figure(figsize=(12, 8))
    plt.barh(plant_names, image_counts, color='skyblue')
    
    # Add labels and title
    plt.xlabel('Number of Images')
    plt.ylabel('Plant Type')
    plt.title('Number of Images per Plant Type')
    
    # Display the plot
    plt.tight_layout()
    plt.savefig('plant_image_distribution.png')
    plt.show()
    print("Distribution graph saved as 'plant_image_distribution.png'")

if __name__ == "__main__":
    dataset_path = "/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation"
    
    # Analyze the dataset
    plant_images = analyze_plant_images(dataset_path)
    
    # Plot the distribution of images per plant
    plot_plant_image_distribution(plant_images)