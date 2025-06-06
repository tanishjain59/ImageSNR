import kagglehub
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download latest version to the data directory
path = kagglehub.dataset_download(
    "matjazmuc/frame-level-driver-drowsiness-detection-fl3d",
    path='data'
)

print("Path to dataset files:", path)