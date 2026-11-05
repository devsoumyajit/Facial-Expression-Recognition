from pathlib import Path
import matplotlib.pyplot as plt

# Dataset paths
train_path = Path("dataset/train")
test_path = Path("dataset/test")

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
image_extensions = (".jpg", ".jpeg", ".png")


# Function to count images in a folder
def count_images(folder_path):

    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    return sum(
        1 for file in folder_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in image_extensions
    )


# Count training and testing images
train_counts = {}
test_counts = {}

for emotion in classes:

    train_counts[emotion] = count_images(
        train_path / emotion
    )

    test_counts[emotion] = count_images(
        test_path / emotion
    )


# Calculate total images
total_train = sum(train_counts.values())
total_test = sum(test_counts.values())


# Display training dataset statistics
print("=" * 65)
print("              DATASET EXPLORATORY ANALYSIS")
print("=" * 65)

print("\nTRAINING DATASET")

for emotion in classes:

    count = train_counts[emotion]
    percentage = (count / total_train) * 100

    print(
        f"{emotion.capitalize():10} : "
        f"{count:5} images ({percentage:5.2f}%)"
    )

print("-" * 65)
print(f"Total training images: {total_train}")


# Display testing dataset statistics
print("\nTESTING DATASET")

for emotion in classes:

    count = test_counts[emotion]
    percentage = (count / total_test) * 100

    print(
        f"{emotion.capitalize():10} : "
        f"{count:5} images ({percentage:5.2f}%)"
    )

print("-" * 65)
print(f"Total testing images: {total_test}")


# Calculate class imbalance ratio
largest_class = max(train_counts.values())
smallest_class = min(train_counts.values())

imbalance_ratio = largest_class / smallest_class

print("\nDATASET SUMMARY")
print(f"Total images: {total_train + total_test}")
print(f"Training images: {total_train}")
print(f"Testing images: {total_test}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")


# Create reports folder
reports_path = Path("reports")
reports_path.mkdir(exist_ok=True)


# Generate class distribution charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training dataset chart
axes[0].bar(
    [emotion.capitalize() for emotion in classes],
    [train_counts[emotion] for emotion in classes]
)

axes[0].set_title("Training Dataset Distribution")
axes[0].set_xlabel("Expression")
axes[0].set_ylabel("Number of Images")
axes[0].tick_params(axis="x", rotation=35)


# Testing dataset chart
axes[1].bar(
    [emotion.capitalize() for emotion in classes],
    [test_counts[emotion] for emotion in classes]
)

axes[1].set_title("Testing Dataset Distribution")
axes[1].set_xlabel("Expression")
axes[1].set_ylabel("Number of Images")
axes[1].tick_params(axis="x", rotation=35)


# Adjust chart layout
plt.tight_layout()

# Save chart
chart_path = reports_path / "class_distribution.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")

print(f"\nChart saved successfully: {chart_path}")

# Display charts
plt.show()