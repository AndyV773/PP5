import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_reg import reg_performance


def page_predict_tomorrows_avg_body():

    # load tenure pipeline files
    version = 'v1'
    tomorrows_avg_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_tomorrows_avg/"
        f"{version}/regressor_pipeline.pkl")
    # tenure_labels_map = load_pkl_file(
    #     f"outputs/ml_pipeline/predict_tenure/{version}/label_map.pkl")
    tomorrows_avg_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_tomorrows_avg/"
        f"{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_tomorrows_avg/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_tomorrows_avg/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_tomorrows_avg/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_tomorrows_avg/{version}/y_test.csv")

    st.write("### ML Pipeline: Forecast Tomorrow's Average Price")
    # display pipeline training summary conclusions
    st.info(
        "* The objective was to develop a **regression model** capable of "
        "accurately predicting **tomorrow's average stock price** based "
        "on historical features. The model's performance exceeded the "
        "project requirements, achieving high predictive accuracy. \n\n"

        "* The evaluation metrics indicate strong performance on both "
        "the train and test sets:\n - **Train Set:** R² Score: **0.992**, "
        "MAE: **6.16**, MSE: **81.88**, RMSE: **9.05**\n - "
        "**Test Set:** R² Score: **0.994**, MAE: **5.96**, "
        "MSE: **71.85**, RMSE: **8.48**\n\n"

        "* These results suggest that the model generalizes well to "
        "unseen data, with minimal error and high correlation between "
        "the predicted and actual average prices. The low **Mean "
        "Absolute Error (MAE)** indicates that the average prediction "
        "error is only around **6 units** for both the training and "
        "testing datasets. Additionally, the **Root Mean Squared Error "
        "(RMSE)**, which penalizes larger errors more heavily, is "
        "consistently low, highlighting stable performance.\n\n"

        "* Overall, the model demonstrates robust predictive capabilities "
        "for forecasting tomorrow's average price, making it a reliable "
        "tool for financial decision-making and risk assessment."
    )

    st.write("---")

    # show pipeline steps
    st.write("* ML pipeline to predict tomorrow's average price")
    st.write(tomorrows_avg_pipe)

    st.write("---")

    # show best features
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(tomorrows_avg_feat_importance)

    st.write("---")

    # evaluate performance on both sets
    st.write("### Pipeline Performance")
    reg_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=tomorrows_avg_pipe)
