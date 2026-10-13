from __init__ import *
from random import uniform, randint
from pprint import pp as print
from bing_image_downloader import downloader

import os
from PIL import Image





def print_(e):
    """Callback function to monitor training loss progression."""
    # e est l'objet Scalar représentant la loss calculée par votre framework
    print(f"Current Loss: {e.data}")




# =====================================================================
# DATA SET MANAGEMENT
# =====================================================================

def download_data():
    """
    Downloads smartphone images directly into the datasets directory.
    Uses query filtering to retrieve relevant brand pictures.
    """
    brands = ["iPhone", "Samsung Galaxy", "Xiaomi phone", "Huawei phone"]

    for brand in brands:
        downloader.download(
            brand, 
            limit=20,  # Number of images to download per brand
            output_dir='datasets/vnet', 
            adult_filter_off=True, 
            force_replace=False, 
            timeout=60
        )

def get_dataset(image_size=16):
    """
    Parses image folders, converts them to grayscale, resizes them,
    and normalizes pixel values strictly between -1.0 and 1.0.
    """
    base_dir = 'datasets/vnet'
    brands = ["iPhone", "Samsung Galaxy", "Xiaomi phone", "Huawei phone"]
    
    X_dataset = []
    Y_dataset = []
    
    # Check if dataset directory exists; if not, trigger automatic download
    if not os.path.exists(base_dir):
        print("Dataset directory not found. Starting automatic download...")
        download_data()

    print("--- Starting image loading and preprocessing pipeline ---")
    for index, brand in enumerate(brands):
        # Assign a unique normalized target value between -1 and 1 for each brand:
        # iPhone = -1.0, Samsung = -0.33, Xiaomi = 0.33, Huawei = 1.0
        target_value = -1.0 + (index * (2.0 / (len(brands) - 1)))
        
        folder_path = os.path.join(base_dir, brand)
        if not os.path.exists(folder_path):
            continue
            
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                with Image.open(img_path) as img:
                    # 1. Convert image to grayscale ('L' mode)
                    img = img.convert('L')
                    # 2. Resize to a lower resolution to optimize CPU computations
                    img = img.resize((image_size, image_size))
                    
                    # 3. Extract raw pixel data and normalize strictly to [-1.0, 1.0]
                    # Since pixel values range from 0 to 255, (p / 127.5) - 1.0 centers data at 0
                    pixels = list(img.getdata())
                    normalized_pixels = [(p / 127.5) - 1.0 for p in pixels]
                    
                    X_dataset.append(normalized_pixels)
                    Y_dataset.append(target_value)
            except Exception:
                # Safely ignore corrupted or unsupported image files
                continue
                
    print(f"Preprocessing complete! {len(X_dataset)} valid images loaded into memory.")
    return X_dataset, Y_dataset

# =====================================================================
# TRAINING & DEMOS
# =====================================================================

def print_(e):
    """Callback function to monitor training loss progression."""
    pass

def run_vnet_demo():
    """Trains the vnet model to classify smartphone brands."""
    # Define image resolution (16x16 pixels = 256 input features)
    img_resolution = 16
    ninput = img_resolution * img_resolution
    
    # Retrieve normalized image dataset
    X, Y = get_dataset(image_size=img_resolution)
    
    if not X:
        print("Error: Could not load the image dataset.")
        return
        
    print("Initializing vnet (Multi-Layer Perceptron architecture)...")
    # Input layer: 256 features, Hidden layers, Output layer: 1 discriminator node
    mlp = MLP(ninput, [16, 4, 1])
    
    
    # Call the native .train() method matching your global framework architecture
    print("Triggering framework native training sequence...")
    mlp.train(x=X, target=Y, callback=print_, lr=0.01, niter=200)

    
    # Run a test forward pass on the very first training sample
    print(f"vnet output prediction: {mlp([X[0]])}  ---  Expected target label: {Y[0]}")

# Execution entry point
if __name__ == "__main__":
    # Execute the smartphone classification task directly
    run_vnet_demo()
