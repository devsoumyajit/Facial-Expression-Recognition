import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

X_train = np.load(os.path.join(DATASET_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATASET_DIR, "y_train.npy"))

X_test = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

print("\nTRAINING LABEL DISTRIBUTION")

for i, name in enumerate(class_names):
    count = np.sum(y_train == i)
    percentage = (count / len(y_train)) * 100
    print(f"{i} = {name:8} : {count:5} images ({percentage:.2f}%)")

print("\nTEST LABEL DISTRIBUTION")

for i, name in enumerate(class_names):
    count = np.sum(y_test == i)
    percentage = (count / len(y_test)) * 100
    print(f"{i} = {name:8} : {count:5} images ({percentage:.2f}%)")

print("\nLABEL VALUES")

print("Training labels:", np.unique(y_train))
print("Testing labels :", np.unique(y_test))

print("\nDATASET CHECK")

print("Training images :", len(X_train))
print("Training labels :", len(y_train))
print("Testing images  :", len(X_test))
print("Testing labels  :", len(y_test))

if len(X_train) == len(y_train) and len(X_test) == len(y_test):
    print("Dataset and labels match correctly.")
else:
    print("ERROR: Dataset and labels do not match.")