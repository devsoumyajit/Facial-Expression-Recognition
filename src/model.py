import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential([
        # Input
        layers.Input(shape=(48, 48, 1)),

        # Convolution Block 1
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 2
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 3
        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten
        layers.Flatten(),

        # Fully Connected Layer
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        # Output: 7 expressions
        layers.Dense(7, activation="softmax")
    ])

    return model


if __name__ == "__main__":
    model = build_model()

    print("FACIAL EXPRESSION CNN MODEL")
    model.summary()