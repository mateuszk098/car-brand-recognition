# Car Brand Recognition

![Static Badge](https://img.shields.io/badge/python-3.12-blue?style=for-the-badge&logo=python&logoColor=white&color=%234584b6)
![GitHub License](https://img.shields.io/github/license/mateuszk098/car-brand-recognition?style=for-the-badge&color=%23fa9537) ![GitHub last commit](https://img.shields.io/github/last-commit/mateuszk098/car-brand-recognition?style=for-the-badge&color=%23fa9537)

## 1. Introduction

This project is a FastAPI-based web application that allows users to upload car images and predict the car brand using a custom Convolutional Neural Network (CNN) model. The model leverages **Residual** and **Squeeze-and-Excitation** connections to enhance prediction performance. It supports predictions for **25 car brands**, and users can specify how many top predictions they would like to see.

## 2. Key Features

- **Custom Deep Learning Model**: Built with lazy layers, including **Residual** and **Squeeze-and-Excitation** blocks, the model can accurately classify cars from a selection of 25 brands.
- **Multiple Car Brands Supported**: Audi, BMW, Toyota, and more...
- **Top K Predictions**: Users can request the top K most probable car brands, allowing for detailed feedback on possible matches.
  
## 3. Authentication and Authorization

- **User Registration & Login**: The application has full authentication and authorization mechanisms in place. Users must create an account and log in before submitting images for prediction.
- **Secure Access**: The app uses OAuth2 with JWT for token-based authentication, ensuring that only authenticated users can access the car brand prediction service.

## 4. How it Works

1. **Upload an Image**: Users can upload an image of a car.
2. **Specify Number of Predictions**: The user can choose how many top predictions they would like to retrieve.
3. **Receive Results**: The app returns a list of predicted car brands with their corresponding probabilities.

## 5. Technology Stack

- **FastAPI**: For building the web API and handling user requests.
- **Custom CNN**: A convolutional neural network enhanced with **Residual** and **Squeeze-and-Excitation** blocks to improve classification performance.
- **Authentication**: OAuth2-based token system for secure user management.
  
## 6. Model Performance Metrics

The following table summarizes the key performance metrics for the car brand classification model:

|              | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|-----------|--------|----------|
| **Value**    | 79.18%   | 80.26%    | 79.18% | 79.25%   |

## 7. Quick Start

1. Go to `car-brand-recognition/` directory.
2. Create `.env` file. It should look as follows:

   ```bash
    # GENERAL
    DEBUG = True

    # MLFLOW POSTGRES
    MLFLOW_POSTGRES_PORT = 5432
    MLFLOW_POSTGRES_USER = "admin"
    MLFLOW_POSTGRES_PASSWORD = "admin"
    MLFLOW_POSTGRES_DB = "mlflow"
    MLFLOW_POSTGRES_HOSTNAME = "mlflow-db"

    # MLFLOW
    MLFLOW_PORT = 5000
    MLFLOW_TRACKING_URI = "http://localhost:${MLFLOW_PORT}"
    MLFLOW_BACKEND_STORE_URI = "postgresql://${MLFLOW_POSTGRES_USER}:${MLFLOW_POSTGRES_PASSWORD}@${MLFLOW_POSTGRES_HOSTNAME}:5432/${MLFLOW_POSTGRES_DB}"

    # PGADMIN
    PGADMIN_PORT = 80
    PGADMIN_DEFAULT_EMAIL = "admin@gmail.com"
    PGADMIN_DEFAULT_PASSWORD = "admin"

    # APP POSTGRES
    APP_POSTGRES_PORT = 5433
    APP_POSTGRES_USER = "admin"
    APP_POSTGRES_PASSWORD = "admin"
    APP_POSTGRES_DB = "brands"
    APP_POSTGRES_HOSTNAME = "app-db"

    # FASTAPI
    MODEL = "SEResNet3"
    BACKEND = "openvino"
    # Create secret key, for example by running: openssl rand -hex 32
    JWT_SECRET_KEY = "d0d88f5cb30b4a4fb0982d3c3210f664f0ed49546bfc1c8ec09f1c9a334a7c47"
    JWT_ALGORITHM = "HS256"
    APP_DB_URL = "postgresql://${APP_POSTGRES_USER}:${APP_POSTGRES_PASSWORD}@${APP_POSTGRES_HOSTNAME}:5432/${APP_POSTGRES_DB}"
    ```

3. Go to `build/` directory and run `docker compose --env-file ../.env up -d`
4. Go to `http://localhost:8000/docs` and voilà.

## 8. Additional Steps

The dataset used for training is available here: https://drive.google.com/file/d/1uzb6maz1DgN2I9W3WESPXGVmO-Wt9qIS/view?usp=sharing
- If you want to experiment with the repository, create a virtual environment and run: `pip install -r requirements.txt`.
- If you don't want to run the application in a container because, for example, you want to experiment with it, comment out `APP_DB_URL` in the `.env` file. Then go to the `app/` directory and run: `fastapi dev main.py` This way, the application will be running directly on the host machine, and every change will be automatically considered.
- If you want to keep an eye on the database, go to `http://localhost:80` and login to pgAdmin 4 using `PGADMIN_DEFAULT_EMAIL` and `PGADMIN_DEFAULT_PASSWORD` that you set in `.env` file. Then, register a new server, you can name it "App". Connection host is `APP_POSTGRES_HOSTNAME` and the password is `APP_POSTGRES_PASSWORD`. If you want to experiment further with the application on the machine host, but subsequently running database on container, change: `{APP_POSTGRES_HOSTNAME}:5432` to `localhost:${APP_POSTGRES_PORT}` in `APP_DB_URL`.
- If you want to trace changes in SQLite database, you can use: `sqlite3 brands.db`, after running the application.
