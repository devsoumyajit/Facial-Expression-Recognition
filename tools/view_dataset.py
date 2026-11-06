import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Dataset training folder
train_path = Path("dataset/train")

# Expression categories
classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Supported image formats
extensions = (".jpg", ".jpeg", ".png")

# Create a figure with 8 spaces (7 images + 1 empty space)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# Convert axes into a simple list
axes = axes.flatten()

# Display one image from each category
for index, emotion in enumerate(classes):

    folder = train_path / emotion

    # Find image files
    images = sorted(
        file for file in folder.iterdir()
        if file.suffix.lower() in extensions
    )

    # Select the first image
    image_path = images[0]

    # Open the image
    image = Image.open(image_path)

    # Display the image
    axes[index].imshow(image.convert("L"), cmap="gray")
    axes[index].set_title(emotion.capitalize())
    axes[index].axis("off")

    # Display image information
    print(f"{emotion.capitalize():10} : {image.size}, Mode: {image.mode}")

# Hide the unused eighth space
axes[7].axis("off")

# Display the images
plt.tight_layout()
plt.show()