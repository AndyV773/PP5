import streamlit as st


def page_summary_body():
    """
    displays the information for the summary page
    """
    st.write("### Quick Project Summary")

    # General project terminology
    st.info("**Project Terms & Jargon**\n"
            "* **stock** refers to the share price of a publicly "
            "traded company\n\n"

            "* **forecast** represents the potential future price of a stock, "
            "estimated based on historical data and market analysis\n\n"

            "* **Market indicators** such as volume, moving averages, "
            "volatility, and momentum are used to understand stock trends\n\n"

            "* The term **target** refers to whether the average stock price "
            "will be higher or lower the next day\n\n"

            "**Project Dataset**\n"
            "* The dataset contains historical stock price data, "
            "including open, close, high, low prices, trading "
            "volume, and technical indicators\n\n"

            "* Each row represents one trading day for a specific stock\n\n"

            "* The goal is to analyze patterns and correlations between "
            "various indicators and the direction of the "
            "next day's stock price")

    # Optional: Link to a GitHub repo or documentation
    st.write("* For additional information, please refer to the "
             "[Project Documentation](https://github.com/AndyV773/PP5).")

    # Business objectives
    st.success("**The project has 4 main business requirements:**\n"
               "* The client wants to uncover key variables or indicators "
               "that are most predictive of whether the stock price will "
               "go up or down the next trading day\n\n"

               "* The client wants to develop a model that can predict "
               "whether the market is likely to move up or down on a "
               "daily basis. This will aid in making more informed "
               "trading decisions and support automated "
               "trading strategies\n\n"

               "* The client also requires a model to predict the daily "
               "average price of the stock, enabling better risk assessment "
               "and the evaluation of potential losses "
               "and market volatility\n\n"

               "* The client requires a dashboard that allows them to "
               "visualize key information, monitor daily predictions, and "
               "interact with data to support day-to-day decision-making")
