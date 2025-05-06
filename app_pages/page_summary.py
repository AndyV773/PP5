import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # General project terminology
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **stock** refers to the share price of a publicly traded company.\n"
        f"* **Market indicators** such as volume, moving averages, volatility, and momentum are used to understand stock trends.\n"
        f"* The term **target variable** refers to whether the stock price will be higher or lower the next day.\n\n"
        
        f"**Project Dataset**\n"
        f"* The dataset contains **historical stock price data**, including open, close, high, low prices, trading volume, and technical indicators.\n"
        f"* Each row represents **one trading day** for a specific stock.\n"
        f"* The goal is to analyze patterns and correlations between various indicators and the direction of the next day’s stock price."
    )

    # Optional: Link to a GitHub repo or documentation
    st.write(
        f"* For additional information, please refer to the "
        f"[Project Documentation](https://github.com/AndyV773/PP5)."
    )

    # Business objectives
    st.success(
        f"The project has 2 main business requirements:\n"
        f"* 1 - The client wants to uncover **key variables or indicators** that are most predictive "
        f"of whether the stock price will go up or down the next trading day.\n"
        f"* 2 - The client aims to develop a model that can provide **daily predictions**, "
        f"enabling more informed decision-making or automated trading strategies."
    )


        