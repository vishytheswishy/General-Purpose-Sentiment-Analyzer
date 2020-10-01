'''
File: evaluation.py
Author: Vishaal Yalamanchali
Purpose: Created to test pickling and importing serialized machine learning models.

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def evaluate(model, X_test, y_test):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))