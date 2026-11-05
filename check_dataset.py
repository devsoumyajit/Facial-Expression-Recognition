import os

# Dataset folder paths
train_path = "dataset/train"
test_path = "dataset/test"

# Seven expression categories
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

# Function to count images
def count_images(folder_path):

    if not os.path.exists(folder_path):
        return None

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith(image_extensions)
    ]

    return len(images)


# Display heading
print("=" * 50)
print("       FACIAL EXPRESSION DATASET CHECK")
print("=" * 50)


# Check training dataset
print("\nTRAINING DATASET")

total_train = 0

for emotion in classes:

    folder = os.path.join(train_path, emotion)

    count = count_images(folder)

    if count is not None:
        total_train += count
        print(f"{emotion.capitalize():10} : {count} images")
    else:
        print(f"{emotion.capitalize():10} : Folder not found")


print("-" * 50)
print(f"Total Training Images : {total_train}")


# Check testing dataset
print("\nTESTING DATASET")

total_test = 0

for emotion in classes:

    folder = os.path.join(test_path, emotion)

    count = count_images(folder)

    if count is not None:
        total_test += count
        print(f"{emotion.capitalize():10} : {count} images")
    else:
        print(f"{emotion.capitalize():10} : Folder not found")


print("-" * 50)
print(f"Total Testing Images  : {total_test}")


# Display overall total
print("\n" + "=" * 50)
print(f"TOTAL DATASET IMAGES   : {total_train + total_test}")
print("=" * 50)

print("\nDataset check completed!")