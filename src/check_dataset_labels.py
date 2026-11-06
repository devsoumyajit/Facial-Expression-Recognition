import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


print("LOADING DATA")

X_train = np.load("dataset/X_train.npy")
y_train = np.load("dataset/y_train.npy")

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")

print("\nCHECKING LABEL VALUES")

print("Unique labels:", np.unique(y_train))

print("\nFIRST 20 LABELS")

for i in range(20):
    print(
        f"Image {i:2d} -> Label {y_train[i]} -> {CLASS_NAMES[y_train[i]]}"
    )


print("\nDISPLAYING SAMPLE IMAGES")

indices = np.random.choice(
    len(X_train),
    size=16,
    replace=False
)

plt.figure(figsize=(10, 10))

for position, index in enumerate(indices):
    plt.subplot(4, 4, position + 1)

    image = X_train[index].squeeze()

    plt.imshow(image, cmap="gray")
    plt.title(
        f"Label: {CLASS_NAMES[y_train[index]]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()