import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib
from preprocess import load_data, preprocess_data
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    return accuracy, f1


