import streamlit as st
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math


# code copied from "Modeling and Evaluation" notebooks
def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)

    # Displaying the evaluation metrics using Streamlit, rounded down
    st.metric(label="R2 Score",
              value=math.floor(r2_score(y, prediction) * 1000) / 1000)
    st.metric(label="Mean Absolute Error",
              value=math.floor(
                  mean_absolute_error(y, prediction) * 1000) / 1000)
    st.metric(label="Mean Squared Error",
              value=math.floor(
                  mean_squared_error(y, prediction) * 1000) / 1000)
    st.metric(label="Root Mean Squared Error",
              value=math.floor(np.sqrt(
                  mean_squared_error(y, prediction)) * 1000) / 1000)


def reg_performance(X_train, y_train, X_test, y_test, pipeline):

    st.write('#### Model Evaluation \n')

    st.info("Train Set")
    regression_evaluation(X_train, y_train, pipeline)

    st.info("Test Set")
    regression_evaluation(X_test, y_test, pipeline)
