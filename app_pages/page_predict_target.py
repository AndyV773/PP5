import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_predict_target_body():
    """
    Displays the "Forecast Target" page in a Streamlit app

    Main Steps:
    - Loads pre-trained ML pipeline, feature importance plot, and datasets

    - Shows model performance metrics, feature importance, "
    "and pipeline structure

    - Optionally displays the first 10 rows of stock data

    - Evaluates model predictions on train and test sets
    """
    version = 'v1'
    # load needed files
    target_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_target/{version}/clf_pipeline_model.pkl")
    target_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_target/"
        F"{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_target/{version}/y_test.csv").values

    st.write("### ML Pipeline: Forecast Target")
    # display pipeline training summary conclusions
    st.info("The pipeline was tuned to achieve at least 0.70 Recall on "
            "the Lower AVG class, as the objective is to minimize "
            "financial risk by accurately identifying downward movements, "
            "while still maintaining an overall accuracy of 0.70 or above\n\n"

            "**The pipeline performance metrics are as follows:**\n\n"

            "* **Train Set:** 0.71 Recall for the 'Lower AVG' class, "
            "with an overall accuracy of 0.70\n\n"

            "* **Test Set:** 0.71 Recall for the 'Lower AVG' class, "
            "with an overall accuracy of 0.70\n\n"

            "While prioritizing recall for risk reduction, the model also "
            "maintains a balanced precision and F1-score, ensuring reliable "
            "predictions across both 'Higher AVG' and 'Lower AVG' classes")

    # show pipelines
    st.write("---")

    st.write("#### There is only 1 ML Pipeline; data "
             "cleaning and feature engineering were not necessary")

    st.write("* The pipline is responsible for feature scaling and modelling")
    st.write(target_pipe_model)

    # show feature importance plot
    st.write("---")

    st.write("* The features the model was trained on and their importance")
    st.write(X_train.columns.to_list())
    st.image(target_feat_importance)

    # evaluate performance on train and test set
    st.write("---")

    st.write("### Pipeline Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=target_pipe_model,
                    label_map=["Lower AVG", "Higher AVG"])
