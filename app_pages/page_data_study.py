import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_management import load_stock_data

sns.set_style("whitegrid")


def page_data_study_body():

    # load data
    df = load_stock_data(1)
    df_clean = load_stock_data(0)

    vars_to_study = ['weekday',
                     'close',
                     'volume']

    st.write("### Data Study")
    st.info(
        "* The client is interested in identifying patterns in historical "
        "stock data to uncover the most relevant variables that influence "
        "whether a stock's value will go up or down the next day."
    )

    # inspect data
    if st.checkbox("Inspect Phoenix Group Holdings "
                   "plc (PHNX 2010 - 2015) Stock Data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.success(
        "* To inspect feature correlations, we extracted "
        "the date into day and year columns, "
        "and added lag features to capture data from up to 2 days prior. "
        "A target variable was also introduced, indicating "
        "whether the stock's value will be higher (1) or lower (0) the "
        "next day. This enables deeper analysis of historical stock patterns "
        "and helps identify key variables that influence price movements."
    )

    # inspect data
    if st.checkbox("Inspect Exploratory Stock Data"):
        st.write(
            f"* The dataset has {df_clean.shape[0]} "
            f"rows and {df_clean.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df_clean.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook "
        f"to better understand how "
        f"the variables are correlated to the target. \n"
        f"Listed below variables are: **{vars_to_study}**"
    )

    # Text based on "02 - Churned Customer Study"
    # notebook - "Conclusions and Next steps" section
    st.info(
        "The correlation indications and plots below interpretation converge. "
        "It is indicated that: \n"
        "* Monday shows the strongest correlation towards target 0, followed "
        "by Tuesday and Wednesday for target 1. However, "
        "the target distribution remains very well balanced, suggesting "
        "low correlation with the target variable. This indicates that the "
        "likelihood of the target being 0 or 1 is relatively stable "
        "and not heavily influenced by the specific day of the week \n\n"
        "* The target for 'open' is very well balanced, indicating that "
        "the opening values do not heavily skew towards one target "
        "class over the other, reflecting market stability or "
        "uniform activity during open periods \n\n"
        "* The target for 'volume' is also balanced, suggesting that "
        "trade or transactional volume does not disproportionately impact "
        "the occurrence of target 0 or 1, implying consistent activity "
        "levels across different trading sessions \n\n"
    )

    # Code copied from "02 - Churned Customer Study"
    # notebook - "EDA on selected variables" section
    df_eda = df_clean.filter(vars_to_study + ['target'])

    # Individual plots per variable
    if st.checkbox("Target Levels per Variable"):
        target_level_per_variable(df_eda)

    st.write("---")

    st.info(
        "The correlation analysis and visualizations below reveal key "
        "insights: \n * There is no significant correlation with volume "
        "or date-related features, suggesting that effective price "
        "predictions require more than just date and volume alone."
    )

    if st.checkbox("Spearman and Pearson Correlation"):

        st.markdown("### Spearman Correlation Heatmap")
        st.image("outputs/heatmaps/spearman_correlation.png")

        st.markdown("### Pearson Correlation Heatmap")
        st.image("outputs/heatmaps/pearson_correlation.png")

    if st.checkbox("PPS Matrix"):

        st.markdown("### Pearson Matrix Heatmap")
        st.image("outputs/heatmaps/pps_matrix.png")


# function created using "02 - Churned Customer Study"
# notebook code - "Variables Distribution by Churn" section
def target_level_per_variable(df_eda):
    target_var = 'target'

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


# code copied from "02 - Churned Customer Study"
# notebook - "Variables Distribution by Churn" section
def plot_categorical(df_clean, col, target_var):
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df_clean, x=col, hue=target_var,
                  order=df_clean[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# code copied from "02 - Churned Customer Study"
# notebook - "Variables Distribution by Churn" section
def plot_numerical(df_clean, col, target_var):
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_clean, x=col, hue=target_var,
                 kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()
