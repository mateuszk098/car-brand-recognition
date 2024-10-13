### Introduction

This project is a FastAPI-based web application that allows users to upload car images and predict the car brand using a custom Convolutional Neural Network (CNN) model. The model leverages **Residual** and **Squeeze-and-Excitation** connections to enhance prediction performance. It supports predictions for **25 car brands**, and users can specify how many top predictions they would like to see.

### Key Features

- **Custom Deep Learning Model**: Built with lazy layers, including **Residual** and **Squeeze-and-Excitation** blocks, the model can accurately classify cars from a selection of 25 brands.
- **Multiple Car Brands Supported**: Audi, BMW, Toyota, and more...
- **Top K Predictions**: Users can request the top K most probable car brands, allowing for detailed feedback on possible matches.
  
### Authentication and Authorization

- **User Registration & Login**: The application has full authentication and authorization mechanisms in place. Users must create an account and log in before submitting images for prediction.
- **Secure Access**: The app uses OAuth2 with JWT for token-based authentication, ensuring that only authenticated users can access the car brand prediction service.

### How it Works

1. **Upload an Image**: Users can upload an image of a car.
2. **Specify Number of Predictions**: The user can choose how many top predictions they would like to retrieve.
3. **Receive Results**: The app returns a list of predicted car brands with their corresponding probabilities.

### Technology Stack

- **FastAPI**: For building the web API and handling user requests.
- **Custom CNN**: A convolutional neural network enhanced with **Residual** and **Squeeze-and-Excitation** blocks to improve classification performance.
- **Authentication**: OAuth2-based token system for secure user management.
  
### Model Performance Metrics

The following table summarizes the key performance metrics for the car brand classification model:

|              | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|-----------|--------|----------|
| **Value**    | 79.18%   | 80.26%    | 79.18% | 79.25%   |
