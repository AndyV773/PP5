import streamlit as st


def page_project_hypothesis_body():
    """
    displays the information for the hypothesis page
    """
    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.error("* **The assumption that correlation patterns between "
             "date or volume and key market indicators, are strong enough "
             "to identify predictive relationships: Incorrect**")

    st.write("* Multiple correlation techniques were applied, including "
             "heatmaps, Predictive Power Score (PPS), and a smart "
             "correlation selection. The smart correlation process "
             "suggested that features such as month, weekday, and high "
             "had some degree of correlation with the target. "
             "However, during model training, feature importance "
             "analysis showed that these variables did not contribute "
             "significantly to predictive performance. Additionally, "
             "the analysis showed that neither volume nor date-related "
             "features exhibited strong enough correlation patterns "
             "or predictive power to support the hypothesis")

    st.success("* **Historical stock data, including key features like "
               "price and volume, can be used in a binary classification "
               "model to predict whether tomorrow's average price will be "
               "higher or lower than today’s, achieving "
               "an accuracy of at least 0.70: Correct**")

    st.write("* The model achieved 0.70 accuracy on the training set and "
             "0.70 accuracy on the test set, confirming that historical "
             "stock data can effectively predict whether tomorrow's "
             "average price will be higher or lower")

    st.success("* **A regression model trained on historical stock data "
               "can accurately forecast tomorrow's average price, and "
               "this forecast can be used to determine the directional "
               "change relative to today’s price: Correct**")

    st.write("* The regression model trained on historical stock data "
             "demonstrates strong predictive performance, achieving "
             "an R² of 0.99 on the test set. This indicates that "
             "0.99 of the variance in the average price is accurately "
             "captured by the model. Additionally, the low MAE "
             "and RMSE confirm precise forecasting with minimal "
             "error")
