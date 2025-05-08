import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_stock_data, load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_predict_target_body():

    df_clean = load_stock_data(0)

    version = 'v1'
    # load needed files
    target_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_target/{version}/clf_pipeline_data_cleaning_feat_eng.pkl')
    target_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_target/{version}/clf_pipeline_model.pkl")
    target_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_target/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/y_test.csv").values

    st.write("### ML Pipeline: Predict Prospect Target")
    # display pipeline training summary conclusions
    st.info(
        f"* The pipeline was tuned aiming at least 0.80 Recall on 'Yes Churn' class, "
        f"since we are interested in this project in detecting a potential churner. \n"
        f"* The pipeline performance on train and test set is 0.90 and 0.85, respectively."
    )

    # inspect data
    if st.checkbox("Inspect Exploratory Stock Data"):
        st.write(
            f"* The dataset has {df_clean.shape[0]} "
            f"rows and {df_clean.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df_clean.head(10))

    # show pipelines
    st.write("---")
    st.write("#### There are 2 ML Pipelines arranged in series.")

    st.write(" * The first is responsible for data cleaning and feature engineering.")
    st.write(target_pipe_dc_fe)

    st.write("* The second is for feature scaling and modelling.")
    st.write(target_pipe_model)

    # show feature importance plot
    st.write("---")
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(target_feat_importance)

    # We don't need to apply dc_fe pipeline, since X_train and X_test
    # were already transformed in the jupyter notebook (Predict Customer Churn.ipynb)

    # evaluate performance on train and test set
    st.write("---")
    st.write("### Pipeline Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=target_pipe_model,
                    label_map=["Lower AVG", "Higher AVG"])
