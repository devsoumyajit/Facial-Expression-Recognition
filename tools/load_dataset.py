import cv2
import numpy as np

from pathlib import Path


# 1. DATASET CONFIGURATION


TRAIN_PATH = Path("dataset/train")
TEST_PATH = Path("dataset/test")

# Seven expression categories
CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Assign a numerical label to each category
CLASS_TO_INDEX = {
    emotion: index
    for index, emotion in enumerate(CLASSES)
}

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Required image dimensions
IMAGE_SIZE = (48, 48)


# 2. LOAD IMAGES FROM A DATASET FOLDER


def load_images(dataset_path):

    images = []
    labels = []

    total_images = 0
    skipped_images = 0

    print(f"\nLoading images from: {dataset_path}")

    for emotion in CLASSES:

        emotion_path = dataset_path / emotion

        if not emotion_path.is_dir():
            raise FileNotFoundError(
                f"Folder not found: {emotion_path}"
            )

        # Get all image files in sorted order
        image_files = sorted(
            file for file in emotion_path.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        emotion_count = 0

        for image_path in image_files:

            # Read image in grayscale
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            # Skip unreadable images
            if image is None:
                skipped_images += 1
                continue

            # Resize image to 48 x 48
            image = cv2.resize(
                image,
                IMAGE_SIZE
            )

            # Normalize pixel values to 0-1
            image = image.astype(np.float32) / 255.0

            # Add channel dimension
            image = np.expand_dims(image, axis=-1)

            # Store image and its numerical label
            images.append(image)
            labels.append(CLASS_TO_INDEX[emotion])

            emotion_count += 1
            total_images += 1

        print(
            f"{emotion.capitalize():10} : "
            f"{emotion_count} images loaded"
        )

    # Convert Python lists into NumPy arrays
    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print(f"\nTotal images loaded: {total_images}")
    print(f"Images skipped: {skipped_images}")

    return X, y


# 3. MAIN PROGRAM


if __name__ == "__main__":

    print("FACIAL EXPRESSION DATASET LOADER")

    # Load training dataset
    X_train, y_train = load_images(TRAIN_PATH)

    # Load testing dataset
    X_test, y_test = load_images(TEST_PATH)

    # Display dataset information

    print("\nDATASET INFORMATION")

    print("TRAINING DATA\n")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Data type:", X_train.dtype)
    print(
        "Pixel range:",
        X_train.min(),
        "to",
        X_train.max()
    )

    print("\nTESTING DATA\n")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Data type:", X_test.dtype)
    print(
        "Pixel range:",
        X_test.min(),
        "to",
        X_test.max()
    )

    # Display label mapping
    print("\nEXPRESSION LABEL MAPPING\n")

    for emotion, label in CLASS_TO_INDEX.items():
        print(f"{label} = {emotion.capitalize()}")

    print("\nDATASET LOADING COMPLETED SUCCESSFULLY!\n")
   
    # Save processed datasets
    np.save("dataset/X_train.npy", X_train)
    np.save("dataset/y_train.npy", y_train)
    np.save("dataset/X_test.npy", X_test)
    np.save("dataset/y_test.npy", y_test)

    print("\nDATASET ARRAYS SAVED SUCCESSFULLY!")