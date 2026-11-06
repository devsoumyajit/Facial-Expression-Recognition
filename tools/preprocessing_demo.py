import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Select one image from the training dataset
image_folder = Path("dataset/train/happy")

# Supported image formats
extensions = (".jpg", ".jpeg", ".png")

# Find the first available image
image_files = sorted(
    file for file in image_folder.iterdir()
    if file.suffix.lower() in extensions
)

if not image_files:
    raise FileNotFoundError("No images found in the selected folder.")

image_path = image_files[0]

# 1. Load image in grayscale
original_image = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

if original_image is None:
    raise ValueError("Unable to read the selected image.")

# 2. Resize image to 48 x 48
resized_image = cv2.resize(
    original_image,
    (48, 48)
)

# 3. Normalize pixel values
normalized_image = resized_image.astype(np.float32) / 255.0

# Display image information

print("IMAGE PREPROCESSING DEMONSTRATION")

print("\nOriginal Image")
print("Shape:", original_image.shape)
print("Data Type:", original_image.dtype)
print("Pixel Range:",
      original_image.min(), "to", original_image.max())

print("\nResized Image")
print("Shape:", resized_image.shape)
print("Data Type:", resized_image.dtype)

print("\nNormalized Image")
print("Shape:", normalized_image.shape)
print("Data Type:", normalized_image.dtype)
print("Pixel Range:",
      normalized_image.min(), "to", normalized_image.max())

# Display all three stages
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original Image")

axes[1].imshow(resized_image, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Resized Image")

axes[2].imshow(normalized_image, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Normalized Image")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()