import streamlit as st


def page_project_hypothesis_body():
    """
    displays the information for the hypothesis page
    """
    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.error("* **The assumption that correlation patterns between "
             "date or volume and key market indicators, are strong enough to "
             "identify predictive relationships: Incorrect**")

    st.write("* Correlation analysis revealed that neither date nor volume "
             "alone demonstrates significant forecasting power for "
             "price movement. These features may require deeper feature "
             "engineering or interaction with other variables "
             "to enhance predictability")

    st.success("* **Historical stock data, including key features like "
               "price and volume, can be used in a binary classification "
               "model to predict whether tomorrow's average price will be "
               "higher or lower than today’s, achieving "
               "an accuracy of at least 0.70: Correct**")

    st.write("* The model achieved 0.71 accuracy on the training set and "
             "0.70 accuracy on the test set, confirming that historical "
             "stock data can effectively predict whether tomorrow's "
             "average price will be higher or lower")

    st.success("* **A regression model trained on historical stock data "
               "can accurately forecast tomorrow's average price, and "
               "this forecast can be used to determine the directional "
               "change relative to today’s price: Correct**")

    st.write("* The regression model trained on historical stock data "
             "demonstrates strong predictive performance, achieving "
             "an R² of 0.997 on the test set. This indicates that "
             "0.997 of the variance in the average price is accurately "
             "captured by the model. Additionally, the low MAE (4.203) "
             "and RMSE (6.115) confirm precise forecasting with minimal "
             "error.")
