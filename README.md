# QuantPlus-Model-Evaluation


## Overview

Building on the QuantPlus dataset, we evaluated classical machine learning models (e.g., KNN, SVM, Random Forest) and deep learning models (e.g., RNN, LSTM, GNN). This project aims to predict individual stock excess returns using 94 predictive cross-sectional features derived from the QuantPlus dataset. The results include model performance metrics such as mean squared error (MSE) and out-of-sample R-squared (R²).

## Data Processing and Dataset Construction

We constructed a dataset consisting of 94 cross-sectional predictive factors to forecast excess returns of individual stocks. Due to the requirements of machine learning and deep learning models, particularly regarding handling missing values, we adopted a consistent preprocessing approach:

### Missing Value Handling: 
Features with a high proportion of missing values were filtered out. For the remaining features, we grouped the data by company name and monthly date, and forward-filled missing values for each company for up to 12 consecutive months (one year).

### Rank Normalization: 
Rank normalization was applied to scale the features appropriately for model training.

The dataset spans 60 years, divided into:

### Training Set: 
Data from 1957 to 1986, containing 1,248,579 data points for model training and validation.

### Test Set: 
Data from 1987 to 2016, containing 2,494,491 data points for out-of-sample testing.

## Model Training Details

### Machine Learning Models

For classical machine learning models, we utilized the highly compatible and widely adopted scikit-learn library to perform regression training. The models were fine-tuned using hyperparameter optimization:

Hyperparameter Tuning: Random Search was applied to select the best hyperparameters. We performed 10 iterations (n_iter=10), with each iteration sampling a different parameter combination and conducting 5-fold cross-validation.

Evaluation Metric: Negative Mean Squared Error (negative MSE) was used as the optimization criterion.

Performance Metrics: After training, we reported the out-of-sample test MSE and R².

### Deep Learning Models

For deep learning models, we used PyTorch, one of the most popular frameworks for building and training models. Each model followed a consistent configuration:

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam optimizer

Learning Rate Scheduler: Fixed learning rate

Maximum Training Epochs: 50

Early Stopping: Patience of 10 epochs

Batch Size: 32

Performance Metrics: After training, we reported the out-of-sample test MSE and R².

### Repository Contents

quantplus data processing.py: Code for data preprocessing, handling missing values, and constructing the dataset

Evaluating classical machine learning models.py: Machine learning model training scripts using scikit-learn.

Evaluating classical deep learning models.py: Deep learning model training scripts using PyTorch.

README.md: This file, providing an overview of the project, data, and model training.
