import streamlit as st


def page_summary_body():

    st.write("### Quick Project Summary")

    # General project terminology
    st.info(
        "**Project Terms & Jargon**\n"
        "* A **stock** refers to the share price of a publicly "
        "traded company.\n"
        "* A **forecast** represents the potential future price of a stock, "
        "estimated based on historical data and market analysis\n"
        "* **Market indicators** such as volume, moving averages, volatility, "
        "and momentum are used to understand stock trends.\n"
        "* The term **target** refers to whether the average stock price "
        "will be higher or lower the next day.\n\n"

        "**Project Dataset**\n"
        "* The dataset contains **historical stock price data**, "
        "including open, close, high, low prices, trading "
        "volume, and technical indicators.\n"
        "* Each row represents **one trading day** for a specific stock.\n"
        "* The goal is to analyze patterns and correlations between various "
        "indicators and the direction of the next day’s stock price."
    )

    # Optional: Link to a GitHub repo or documentation
    st.write(
        "* For additional information, please refer to the "
        "[Project Documentation](https://github.com/AndyV773/PP5)."
    )

    # Business objectives
    st.success(
        "The project has 3 main business requirements:\n"
        "* The client wants to uncover key variables or indicators "
        "that are most predictive of whether the stock price will "
        "go up or down the next trading day.\n\n"
        "* The client is looking to have a model developed that can "
        "generate daily predictions, enabling more informed "
        "decision-making, supporting automated trading strategies, "
        "and incorporating risk assessment to evaluate potential "
        "losses and market volatility. \n\n"
        "* The client requires a dashboard that allows them to "
        "visualize key information, monitor daily predictions, and "
        "interact with data to support day-to-day decision-making"
    )
