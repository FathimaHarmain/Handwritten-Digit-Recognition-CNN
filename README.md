
# DigitVision - Handwritten Digit Recognition

## Project Overview

DigitVision is a deep-learning application that recognizes
handwritten numerical digits from images using a
Convolutional Neural Network (CNN).

The model is trained on the MNIST dataset and classifies
handwritten digits into ten categories (0-9).

## Features

- Handwritten digit image upload
- Image preprocessing and normalization
- CNN-based digit classification
- Prediction confidence scores
- Visualization of prediction scores
- Interactive Streamlit application

## Dataset

MNIST Handwritten Digit Dataset

https://www.tensorflow.org/datasets/catalog/mnist

The dataset contains 60,000 training images and
10,000 testing images.

## Technologies Used

- Python
- TensorFlow
- Keras
- CNN
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn
- Streamlit

## Model Architecture

- Convolutional layers
- Batch normalization
- Max pooling
- ReLU activation
- Dropout
- Fully connected layers
- Softmax output layer

## Installation

pip install -r requirements.txt

## Run Application

streamlit run app.py

## Model File

handwritten_digit_cnn.keras

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Project Structure

app.py
handwritten_digit_cnn.keras
evaluation_metrics.json
requirements.txt
README.md
