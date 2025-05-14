import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# code copied from "Modeling and Evaluation" notebooks
def confusion_matrix_and_report(X, y, pipeline, label_map):
    """
    Evaluate the performance of classification model
    and display metrics

    Parameters:
    - X: Features of the dataset
    - y: True values of the target variable
    - pipeline: Trained model pipeline to make predictions

    Displays:
    - Precision
    - Recall
    - F1-score
    - Support
    """
    prediction = pipeline.predict(X)

    st.write('#### Confusion Matrix')
    st.code(pd.DataFrame(confusion_matrix(y_true=prediction, y_pred=y),
                         columns=[["Actual " + sub for sub in label_map]],
                         index=[["Prediction " + sub for sub in label_map]]))

    st.write('#### Classification Report')
    st.code(classification_report(y, prediction, target_names=label_map), "\n")


# code copied from "Modeling and Evaluation" notebooks
def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    """
    Displays classification metrics for the model on both
    training and test datasets
    """
    st.info("Train Set")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    st.info("Test Set")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)
