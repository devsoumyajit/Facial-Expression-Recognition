import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model import build_model

# 1. LOAD DATASET

print("LOADING DATASET")

X_train = np.load("dataset/X_train.npy")
y_train = np.load("dataset/y_train.npy")

X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_test shape  : {y_test.shape}")

# 2. CHECK LABELS

print("\nCHECKING LABELS")

print(f"Training labels: {np.unique(y_train)}")
print(f"Testing labels : {np.unique(y_test)}")

# 3. CREATE VALIDATION SET

print("\nCREATING VALIDATION SET")

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Testing samples    : {len(X_test)}")


# 4. BUILD MODEL

print("\nBUILDING CNN MODEL")

model = build_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. CREATE OUTPUT FOLDERS

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 6. TRAIN MODEL

print("\nSTARTING CNN TRAINING")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    shuffle=True,
    verbose=1
)


# 7. EVALUATE ON TEST DATA

print("\nEVALUATING MODEL")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n======================================")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print("======================================")


# 8. SAVE MODEL

model.save("models/expression_cnn.keras")

print("\nModel saved successfully")
print("Location: models/expression_cnn.keras")


# 9. SAVE TRAINING HISTORY

np.save(
    "reports/training_accuracy.npy",
    history.history["accuracy"]
)

np.save(
    "reports/validation_accuracy.npy",
    history.history["val_accuracy"]
)

np.save(
    "reports/training_loss.npy",
    history.history["loss"]
)

np.save(
    "reports/validation_loss.npy",
    history.history["val_loss"]
)

print("Training history saved successfully")

print("\nTRAINING COMPLETED")