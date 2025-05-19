import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_management import load_stock_data

sns.set_style("whitegrid")


def page_data_study_body():
    """
    Displays Streamlit interface for analyzing historical stock data,
    identifying key features correlated with predicting stock price movements

    Main Features:
    - Data Inspection: Displays raw data and exploratory data
    - Correlation Analysis: Identifies relationships between
    features and the target
    - Visual Analysis: Renders heatmaps and distribution plots
    for deeper insights

    Target Variables:
    - 'month': Correlation with stock price increases in specific months
    - 'weekday': Trends in stock performance on specific days
    - 'high': Consistent price peaks without bias
    """
    # load data
    df = load_stock_data(1)
    df_clean = load_stock_data(0)

    vars_to_study = ['month',
                     'weekday',
                     'high']

    vars_to_plot = ['year',
                    'month',
                    'weekday',
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume',
                    'average']

    st.write("### Data Study")
    st.info("* The client is interested in identifying patterns in historical "
            "stock data to uncover the most relevant market indicators that "
            "are correlated with forecasting the target")

    # inspect data
    if st.checkbox("Inspect Phoenix Group Holdings "
                   "plc (PHNX 2010 - 2025) Stock Data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns")

        st.write(df)

    st.success("* To analyze feature correlations, we extracted "
               "the date into day, month, and year columns, "
               "and added lag features to capture data from up to 2 days "
               "prior. A target variable was also introduced, indicating "
               "whether the stock's value will be 1 (higher) or 0 (lower) "
               "the next day. This enables deeper exploration of "
               "historical stock patterns and helps identify "
               "key variables that influence price movements")

    # inspect data
    if st.checkbox("Inspect Exploratory Stock Data"):
        st.write(
            f"* The dataset has {df_clean.shape[0]} "
            f"rows and {df_clean.shape[1]} columns")

        st.write(df_clean)

    st.write("---")

    # inspect charts
    st.write("The stock price has remained within a relatively "
             "narrow range, with a noticeable increase between "
             "2015 and 2021 before returning to its previous levels. In "
             "contrast, trading volume has shown a significant "
             "upward trend during this time")

    if st.checkbox("Stock Charts"):

        st.info("* The line chart provides a clear visualization of stock "
                "price movements over the years, highlighting key trends, "
                "fluctuations, and long-term performance")
        st.markdown("### Line Chart")
        st.image("docs/plots/line_plot.png")

        st.info("* The bar graph provides a clear visualization of the "
                "highs and lows in market activity and trading volume "
                "highlighting significant increases that indicates "
                "long-term growth")
        st.markdown("### Bar Graph")
        st.image("docs/plots/volume_plot.png")

    st.write("---")

    # Correlation Study Summary
    st.write(f"Smart correlation selection was performed in the "
             f"notebook to identify variables with the strongest "
             f"relationships to the target\n\n "
             f"The selected variables are: **{vars_to_study}**")

    # Text based on "02 - Churned Customer Study"
    # notebook - "Conclusions and Next steps" section
    st.info("**The correlation indications and plots below interpretation "
            "converge. It is indicated that:**\n\n"

            "* November shows a stronger correlation towards target 1, "
            "while September leans more towards target 0. Despite these "
            "monthly tendencies, the overall target distribution remains "
            "relatively balanced, indicating that while certain months "
            "exhibit slight biases, the likelihood of the target being "
            "0 or 1 is generally stable throughout the year \n\n"

            "* Monday shows the strongest correlation towards target 0, "
            "followed by Tuesday and Wednesday for target 1. However, "
            "the target distribution remains very well balanced, suggesting "
            "low correlation with the target variable. This indicates that "
            "the likelihood of the target being 0 or 1 is relatively stable "
            "and not heavily influenced by the specific day of the week \n\n"

            "* The target for 'high' is also balanced, indicating that the "
            "maximum daily prices are not skewed towards target 0 or 1. "
            "This suggests consistent price peaks across different trading "
            "sessions, without significant bias towards one class "
            "over the other \n\n")

    # Code copied from "02 - Churned Customer Study"
    # notebook - "EDA on selected variables" section
    df_eda = df_clean.filter(vars_to_study + ['target'])

    # Individual plots per variable
    if st.checkbox("Target Levels per Variable"):
        target_level_per_variable(df_eda)

    st.write("---")

    st.info("**The correlation analysis and visualizations below reveal key "
            "insights:**\n * There is no significant correlation with volume "
            "or date-related features, suggesting that effective price "
            "predictions require more than just date and volume alone")

    if st.checkbox("Spearman and Pearson Correlation"):

        st.markdown("### Spearman Correlation Heatmap")
        st.image("docs/plots/spearman_corr.png")

        st.markdown("### Pearson Correlation Heatmap")
        st.image("docs/plots/pearson_corr.png")

    if st.checkbox("PPS Matrix"):

        st.write("* While the PPS heatmap highlights strong predictive "
                 "relationships with the 'year' feature and key market "
                 "indicators, the latter is too high-level to "
                 "meaningfully contribute to market trend forecasts. "
                 "Its lack of granularity prevents it from capturing "
                 "short-term or structural market dynamics")

        st.markdown("### Pearson Matrix Heatmap")
        st.image("docs/plots/pps_matrix.png")

    st.write("---")

    df_plot = df_clean.filter(vars_to_plot + ['target'])

    st.write("The plots below display each variable plotted against "
             "the target. Classes 0 and 1 are well balanced, which "
             "does not allow for effective predictive analysis over a "
             "given variable, as there is no clear dominance that "
             "could indicate stronger predictive power")

    variable = st.selectbox('Select a feature to visualize correlation:',
                            df_plot.columns[:-1])

    target_level_per_variable_custom(df_plot, variable)


def target_level_per_variable_custom(df_plot, variable):
    """
    Plots the distribution of the selected variable from the DataFrame,
    segmented by the target variable. Categorical features are displayed
    using count plots, and numerical features are displayed using histograms
    """
    target_var = 'target'

    if df_plot[variable].dtype == 'object':
        plot_categorical(df_plot, variable, target_var)
    else:
        plot_numerical(df_plot, variable, target_var)


# function created using "02 - Churned Customer Study"
# notebook code - "Variables Distribution by Churn" section
def target_level_per_variable(df_eda):
    """
    Plots the distribution of each variable in the DataFrame,
    segmented by the target variable. Categorical features are displayed
    using count plots, and numerical features are displayed using histograms
    """
    target_var = 'target'

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


# code copied from "02 - Churned Customer Study"
# notebook - "Variables Distribution by Churn" section
def plot_categorical(df_clean, col, target_var):
    """
    Generates a count plot for a categorical variable,
    segmented by the target variable
    """
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df_clean, x=col, hue=target_var,
                  order=df_clean[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# code copied from "02 - Churned Customer Study"
# notebook - "Variables Distribution by Churn" section
def plot_numerical(df_clean, col, target_var):
    """
    Generates a histogram with a KDE overlay for a numerical
    variable, segmented by the target variable
    """
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_clean, x=col, hue=target_var,
                 kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()
