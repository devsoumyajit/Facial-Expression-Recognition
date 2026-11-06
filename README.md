# Real-Time Facial Expression Recognition and Analysis System Using Deep Learning

A real-time facial expression recognition system that uses **Deep Learning and Computer Vision** to detect human faces and classify visible facial expressions into seven categories.

The system supports **image-based recognition, real-time webcam recognition, session recording, and analytics** through a Streamlit dashboard.

---

## 1. Project Overview

Facial expressions provide important information about a person's visible emotional state. This project aims to build a computer-based system that can detect a face from an image or webcam stream and classify the visible facial expression using a Convolutional Neural Network (CNN).

The system does **not identify a person's identity**. It identifies the **facial expression shown by the detected face**.

### Example

```
Input Image / Webcam
        ↓
Face Detection
        ↓
Face Cropping
        ↓
Grayscale Conversion
        ↓
Resize to 48 × 48
        ↓
Normalization
        ↓
CNN Model
        ↓
Expression Classification
        ↓
Expression + Confidence
        ↓
Dashboard / Analytics
```

---

# 2. Features

- Real-time facial expression recognition
- Face detection using OpenCV Haar Cascade
- CNN-based expression classification
- Image-based expression prediction
- USB webcam / DroidCam support
- Detection of multiple faces
- Expression confidence score
- Expression smoothing for more stable real-time results
- Session-based prediction recording
- SQLite database storage
- Session analytics
- Expression distribution analysis
- Confidence analysis
- Face detection statistics
- Streamlit dashboard
- Separate live recognition and analytics pages

---

# 3. Supported Facial Expressions

The system recognizes **7 facial expression classes**:

| Class |  Expression  |
| ----- | ------------ |
|   0   |   Angry      |
|   1   |   Disgust    |
|   2   |   Fear       |
|   3   |   Happy      |
|   4   |   Neutral    |
|   5   |   Sad        |
|   6   |   Surprise   |

---

# 4. System Architecture

```text
                 INPUT
          ┌─────────────────┐
          │ Image / Webcam  │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │ Face Detection  │
          │ OpenCV Haar     │
          │ Cascade         │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │ Face Cropping   │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │ Preprocessing   │
          │ Grayscale       │
          │ 48 × 48         │
          │ Normalize       │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │ CNN Model       │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │ 7-Class         │
          │ Classification  │
          └────────┬────────┘
                   ↓
       ┌───────────┴────────────┐
       ↓                        ↓
 Expression +             SQLite Database
 Confidence                    ↓
       ↓                  Session Analytics
       ↓                        ↓
 Streamlit Live          Streamlit Dashboard
 Recognition
```

---

# 5. Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- Convolutional Neural Network (CNN)

### Computer Vision

- OpenCV
- Haar Cascade Classifier

### Data Processing

- NumPy
- Pandas

### Machine Learning Utilities

- Scikit-learn

### Data Visualization

- Matplotlib
- Plotly

### Web Application

- Streamlit

### Database

- SQLite

### Version Control

- Git
- GitHub
- Git LFS

---

# 6. Dataset

The project uses a facial expression image dataset containing **35,887 grayscale images**.

### Dataset split

| Dataset | Number of Images |
|---|---:|
| Training | 28,709 |
| Testing | 7,178 |
| Total | 35,887 |

### Image properties

```text
Image size : 48 × 48 pixels
Image type : Grayscale
Classes    : 7
```

### Dataset structure

```text
dataset/
├── train/
├── test/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

The dataset contains images belonging to the following classes:

```text
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

---

# 7. Data Preprocessing

Before an image is passed to the CNN, the following preprocessing steps are performed:

```text
Input Face
    ↓
Grayscale
    ↓
Resize to 48 × 48
    ↓
Convert to float32
    ↓
Normalize pixel values
    ↓
Pixel values / 255
    ↓
CNN Input
```

The final input shape to the model is:

```text
(48, 48, 1)
```

---

# 8. CNN Model Architecture

The project uses a custom Convolutional Neural Network.

### Main architecture

```text
Input
48 × 48 × 1

        ↓

Conv2D
32 filters
3 × 3
ReLU

        ↓

Batch Normalization

        ↓

Max Pooling
2 × 2

        ↓

Conv2D
64 filters
3 × 3
ReLU

        ↓

Batch Normalization

        ↓

Max Pooling
2 × 2

        ↓

Conv2D
128 filters
3 × 3
ReLU

        ↓

Batch Normalization

        ↓

Max Pooling
2 × 2

        ↓

Flatten

        ↓

Dense
128 neurons
ReLU

        ↓

Dropout
0.5

        ↓

Dense
7 neurons
Softmax

        ↓

Expression Prediction
```

The final Softmax layer produces probabilities for all seven expressions.

---

# 9. Model Training

The model is trained using:

```text
Optimizer  : Adam
Learning Rate : 0.001
Loss       : Sparse Categorical Crossentropy
Metric     : Accuracy
Batch Size : 64
Epochs     : Up to 30
```

Training uses:

- Training data
- Validation split
- Early stopping
- Model checkpointing
- Learning-rate reduction

The best model is saved as:

```text
models/expression_cnn.keras
```

---

# 10. Baseline Model Result

The current baseline CNN achieved approximately:

```text
Test Accuracy : 54.25%
```

The training accuracy was higher than the validation/test accuracy, indicating that the current model shows **overfitting**.

Therefore, the current result is treated as a baseline rather than the final optimized performance.

Future model improvements can focus on:

- Data augmentation
- Improved CNN architecture
- Class balancing
- Regularization
- Hyperparameter tuning
- Transfer learning
- Improved preprocessing

---

# 11. Face Detection

Face detection is performed using the OpenCV Haar Cascade classifier.

The cascade file is stored at:

```text
src/haarcascade_frontalface_default.xml
```

The detector identifies face regions before the CNN performs expression classification.

```text
Full Frame
    ↓
Haar Cascade
    ↓
Face Bounding Box
    ↓
Face Crop
    ↓
CNN Prediction
```

---

# 12. Image-Based Recognition

The project supports expression recognition from individual images.

Example command:

```powershell
python src/predict.py test.jpg
```

The system:

1. Loads the trained CNN model.
2. Loads the input image.
3. Converts it to grayscale.
4. Resizes it to 48 × 48.
5. Normalizes the image.
6. Sends it to the CNN.
7. Calculates expression probabilities.
8. Displays the predicted expression and confidence.

---

# 13. Combined Face Detection and Expression Recognition

The project also supports detecting faces and predicting expressions from the same image.

Example:

```powershell
python src/expression_recognition.py test2.jpg
```

The system draws a bounding box around detected faces and displays:

```text
Expression
Confidence
```

The processed result is saved inside:

```text
outputs/
```

---

# 14. Real-Time Recognition

The project supports real-time recognition through a webcam.

A USB-connected DroidCam can also be used as the camera source.

The real-time pipeline is:

```text
USB Camera
     ↓
OpenCV
     ↓
Face Detection
     ↓
Face Crop
     ↓
CNN Prediction
     ↓
Expression Smoothing
     ↓
Expression + Confidence
     ↓
SQLite Logging
```

To start the Streamlit application:

```powershell
streamlit run app.py
```

Then open the Streamlit application in the browser.

The **Live Recognition** page provides:

- Start Camera
- Stop Camera
- Real-time face detection
- Expression prediction
- Confidence display
- Session recording

---

# 15. Prediction Smoothing

Real-time CNN predictions can change from frame to frame.

To make the displayed prediction more stable, the system processes selected frames and maintains recent prediction history.

Instead of immediately displaying every single prediction, recent predictions are considered together.

This helps reduce rapid changes such as:

```text
Happy → Neutral → Happy → Neutral
```

and produces a more stable displayed result.

---

# 16. SQLite Database

Prediction information is stored using SQLite.

Database location:

```text
database/expression_recognition.db
```

The system records information such as:

- Prediction ID
- Session ID
- Timestamp
- Expression
- Confidence
- Number of detected faces

Example:

```text
ID: 1
Session ID: TEST_SESSION_001
Timestamp: 2026-11-07 03:06:42
Expression: Happy
Confidence: 92.50%
Faces detected: 1
```

---

# 17. Session Analytics

The system groups predictions into sessions.

Analytics include:

- Total predictions
- Total sessions
- Average confidence
- Average faces detected
- Most detected expression
- Expression distribution
- Session-wise prediction count
- Session-wise average confidence

Example expression distribution:

```text
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

---

# 18. Streamlit Dashboard

The Streamlit application contains separate pages for the major system functions.

### Live Recognition

Provides:

- Camera control
- Real-time face detection
- Expression prediction
- Confidence display
- Session logging

### Session Analytics

Provides:

- Overall metrics
- Session filtering
- Expression distribution
- Confidence analysis
- Face detection analysis
- Session history
- Selected session details
- Prediction records

---

# 19. Project Structure

```text
Facial_Expression_Recognition/
│
├── .gitignore
├── requirements.txt
├── README.md
├── app.py
│
├── dataset/
│   ├── train/
│   ├── test/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   └── y_test.npy
│
├── models/
│   └── expression_cnn.keras
│
├── reports/
│   ├── class_distribution.png
│   ├── training_accuracy.npy
│   ├── validation_accuracy.npy
│   ├── training_loss.npy
│   ├── validation_loss.npy
│   └── learning_rate.npy
│
├── database/
│   └── expression_recognition.db
│
├── pages/
│   ├── 1_Live_Recognition.py
│   └── 2_Session_Analytics.py
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── check_labels.py
│   ├── check_dataset_labels.py
│   ├── face_detector.py
│   ├── expression_recognition.py
│   ├── webcam_recognition.py
│   ├── database.py
│   ├── live_recognition.py
│   └── session_analytics.py
│
└── tools/
    ├── analyze_dataset.py
    ├── check_dataset.py
    ├── load_dataset.py
    ├── preprocessing_demo.py
    └── view_dataset.py
```

---

# 20. Installation

Clone the repository:

```powershell
git clone https://github.com/devsoumyajit/Facial-Expression-Recognition.git
```

Move into the project directory:

```powershell
cd Facial-Expression-Recognition
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 21. Running the Project

The project can be used immediately if the trained model is already present.

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application provides:

```text
Live Recognition
Session Analytics
```

---

# 22. Training the Model

To train the CNN from the included dataset:

```powershell
python src/train.py
```

The training script:

1. Loads the dataset.
2. Creates a validation set.
3. Builds the CNN.
4. Compiles the model.
5. Trains the model.
6. Performs validation.
7. Evaluates the test dataset.
8. Saves the best model.
9. Saves training history.

The trained model is saved to:

```text
models/expression_cnn.keras
```

---

# 23. Evaluating the Dataset

Dataset labels can be checked using:

```powershell
python src/check_labels.py
```

The script verifies:

- Training samples
- Testing samples
- Label values
- Class distribution
- Image/label matching

---

# 24. GitHub and Git LFS

The project uses **Git LFS** for large NumPy dataset files.

This is necessary because the dataset arrays are much larger than normal GitHub file sizes.

Tracked files include:

```text
*.npy
```

After making changes:

```powershell
git add .
git commit -m "Description of changes"
git push
```

Check repository status:

```powershell
git status
```

---

# 25. Future Improvements

The project can be further improved by:

- Increasing model accuracy
- Reducing overfitting
- Applying data augmentation
- Handling class imbalance
- Improving facial preprocessing
- Experimenting with deeper CNN architectures
- Trying transfer learning
- Adding more detailed evaluation metrics
- Improving real-time performance
- Adding graphical reports
- Improving session analytics
- Adding exportable analytics reports

---

# 26. Limitations

The current system has some limitations:

- Facial expression recognition is affected by lighting conditions.
- Face angle and occlusion can affect detection.
- Expressions can be ambiguous.
- The current baseline CNN shows overfitting.
- The model's confidence score represents model probability and should not be interpreted as a guaranteed measurement of a person's actual emotional state.
- The system classifies visible facial expressions and does not determine a person's true internal emotional state.

---

# 27. Disclaimer

This project is intended for **academic and educational purposes**.

The system recognizes visible facial expression patterns from images or video. It should not be used to make sensitive decisions about a person's mental state, personality, intentions, or psychological condition.

---

# 28. Project Repository

GitHub repository:

https://github.com/devsoumyajit/Facial-Expression-Recognition

---

## Conclusion

This project demonstrates the integration of:

```text
Computer Vision
        +
Deep Learning
        +
CNN Classification
        +
Real-Time Processing
        +
SQLite
        +
Streamlit
        +
Data Analytics
```

The final system provides an end-to-end workflow from **face detection and expression classification to real-time visualization and session analytics**.