import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import time
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning
import sklearn


# Ignore DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
print(sklearn.__version__)


def calculate_r2_oos(real, predict):
    real = np.array(real)
    predict = np.array(predict)
    assert real.shape == predict.shape
    valid_mask = ~np.isnan(real)
    numerator = np.sum((real[valid_mask] - predict[valid_mask]) ** 2)
    denominator = np.sum(real[valid_mask] ** 2)
    R2_oos = 1 - (numerator / denominator)
    return R2_oos


def train_model(train_loader, test_loader, model_type='adaboost', task='classification', n_iter=5, save_path=None):
    # Convert DataLoader to numpy arrays and flatten y
    X_train = np.vstack([x.numpy() for x, y in tqdm(train_loader, desc="Processing Training Data")])
    y_train = np.concatenate([y.numpy() for x, y in train_loader]).ravel()
    X_test = np.vstack([x.numpy() for x, y in tqdm(test_loader, desc="Processing Testing Data")])
    y_test = np.concatenate([y.numpy() for x, y in test_loader]).ravel()

    # Select model and parameters based on task and model type
    if task == 'classification':
        if model_type == 'adaboost':
            model = AdaBoostClassifier(estimator=DecisionTreeClassifier())
            param_distributions = {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.1, 1, 10],
                'estimator__max_depth': [1, 2, 3, 4, 5]
            }
        elif model_type == 'knn':
            model = KNeighborsClassifier()
            param_distributions = {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
        elif model_type == 'random_forest':
            model = RandomForestClassifier()
            param_distributions = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'criterion': ['gini', 'entropy']
            }
        elif model_type == 'naive_bayes':
            model = GaussianNB()
            param_distributions = {}  # No hyperparameters to tune for Naive Bayes
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier()
            param_distributions = {
                'max_depth': [None, 10, 20, 30],
                'criterion': ['gini', 'entropy']
            }
        else:
            raise ValueError("Unsupported model type")
    else:
        if model_type == 'adaboost':
            model = AdaBoostRegressor(estimator=DecisionTreeRegressor())
            param_distributions = {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.1, 1, 10],
                'estimator__max_depth': [1, 2, 3, 4, 5]
            }
        elif model_type == 'knn':
            model = KNeighborsRegressor()
            param_distributions = {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
        elif model_type == 'random_forest':
            model = RandomForestRegressor()
            param_distributions = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30]
            }
        elif model_type == 'decision_tree':
            model = DecisionTreeRegressor()
            param_distributions = {
                'max_depth': [None, 10, 20, 30]
            }
        else:
            raise ValueError("Unsupported model type")

    # Use RandomizedSearchCV for hyperparameter tuning
    search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=n_iter, cv=3, n_jobs=-1, verbose=3)

    start_time = time.time()
    print(f"Starting {model_type} model training and hyperparameter search...")

    # Train model with hyperparameter tuning
    search.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Get the best model
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate model performance
    if task == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{model_type.capitalize()} Best Accuracy: {accuracy:.4f}")
        print(f"{model_type.capitalize()} Precision: {precision:.4f}")
        print(f"{model_type.capitalize()} Recall: {recall:.4f}")
        print(f"{model_type.capitalize()} F1 Score: {f1:.4f}")

        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_params': search.best_params_
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2_oos = calculate_r2_oos(y_test, y_pred)

        print(f"\n{model_type.capitalize()} Best MSE: {mse:.4f}")
        print(f"{model_type.capitalize()} Out-of-Sample R²: {r2_oos:.4f}")

        result = {
            'mse': mse,
            'r2_oos': r2_oos,
            'best_params': search.best_params_
        }

    # Save model and results if save_path is specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, f"{model_type}_{task}.joblib")
        result_save_path = os.path.join(save_path, f"{model_type}_{task}_results.txt")
        joblib.dump(best_model, model_save_path)
        with open(result_save_path, 'w') as f:
            f.write(f"Best Model Parameters: {search.best_params_}\n")
            f.write(f"Validation Results: {result}\n")
        print(f"Model and results saved to {save_path}")