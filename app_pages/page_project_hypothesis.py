import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.error("* **The assumption that correlation patterns between "
             "date or volume and key stock indicators (such as open, "
             "close, and other related metrics) are strong enough to "
             "identify predictive relationships is incorrect.**")
    st.write("* Correlation analysis revealed that neither date nor volume "
             "alone demonstrates significant predictive power for "
             "price movement. These features may require deeper feature "
             "engineering or interaction with other variables "
             "to enhance predictability.")

    st.success("* **Historical stock data, including key features like "
               "price and volume, can be used in a binary classification "
               "model to predict whether tomorrow's average price will be "
               "higher or lower than today’s, achieving "
               "an accuracy of at least 70%.**")

    st.success("* **A regression model trained on historical stock data "
               "(open, close, and volume) can accurately forecast "
               "tomorrow's average price, and this forecast can be used to "
               "determine the directional change relative "
               "to today’s price.**")

    st.success("* **Unsupervised learning techniques, such as clustering, "
               "can identify distinct market regimes or patterns in "
               "historical price and volume data. These insights can "
               "enhance the prediction of future price directions and "
               "help identify significant market structures.**")
