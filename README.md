# Emotion Detection Using Keras and OpenCV
### Overview
This project demonstrates real-time emotion detection using a pre-trained Convolutional Neural Network (CNN). The model is trained using the Keras framework and then used to classify facial expressions into one of seven emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

### Files
TrainEmotionDetector.py:
    Used to build and train the emotion detection model using Keras.
    Saves the model architecture in emotion_model.json and weights in emotion_model.h5.
    Utilizes the FER-2013 dataset to train the CNN.

EmotionDetector.py:
    Loads the pre-trained model and its weights.
    Detects faces from a video or webcam feed using Haar cascades.
    Predicts the emotion for each detected face and displays the result in real time.

# Project Structure
.
├── data/
│   ├── train/           # Training dataset
│   └── test/            # Validation dataset
├── model/
│   ├── emotion_model.json
│   └── emotion_model.h5
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
├── TrainEmotionDetector.py
├── EmotionDetector.py
└── README.md

# Requirements
Python 3.8+
TensorFlow/Keras
OpenCV
NumPy
Matplotlib
Scikit-learn

# Training the Model
Run the following script to train the model:
python TrainEmotionDetector.py

This will:
Load and preprocess the training and validation datasets.
Build and train a CNN for emotion classification.
Save the model architecture (emotion_model.json) and weights (emotion_model.h5).

# Running Emotion Detection
Run the following script to detect emotions in real-time:
python EmotionDetector.py

This will:
Load the pre-trained model.
Open a video stream or use a provided video file.
Detect faces in the frames and predict emotions.
Display the results in real-time.

# Notes
Ensure the haarcascade_frontalface_default.xml file is present in the haarcascades/ folder for face detection.
Replace the video path in EmotionDetector.py with your video file or uncomment the webcam line for real-time detection.
Results
The model detects and classifies facial expressions with reasonable accuracy.
A confusion matrix and classification report can be generated in TrainEmotionDetector.py for performance evaluation."# Emotion-Detector" 
