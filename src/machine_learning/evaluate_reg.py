import streamlit as st
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# code copied from "Modeling and Evaluation" notebooks
def regression_evaluation(X, y, pipeline):
    """
    Evaluate the performance of a regression model
    and display metrics in Streamlit

    Parameters:
    - X: Features of the dataset
    - y: True values of the target variable
    - pipeline: Trained model pipeline to make predictions

    Displays:
    - R2 Score
    - Mean Absolute Error
    - Mean Squared Error
    - Root Mean Squared Error
    """
    prediction = pipeline.predict(X)

    r2 = np.round(r2_score(y, prediction), 3)
    mae = np.round(mean_absolute_error(y, prediction), 3)
    mse = np.round(mean_squared_error(y, prediction), 3)
    rmse = np.round(np.sqrt(mean_squared_error(y, prediction)), 3)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="R2 Score", value=r2)
        st.metric(label="Mean Absolute Error", value=mae)
    with col2:
        st.metric(label="Mean Squared Error", value=mse)
        st.metric(label="Root Mean Squared Error", value=rmse)


def reg_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Displays regression metrics for the model on both
    training and test datasets
    """
    st.write('#### Model Evaluation \n')

    st.info("Train Set")
    regression_evaluation(X_train, y_train, pipeline)

    st.info("Test Set")
    regression_evaluation(X_test, y_test, pipeline)
