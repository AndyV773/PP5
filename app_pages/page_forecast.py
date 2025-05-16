import streamlit as st
import pandas as pd
import itertools
from src.data_management import load_stock_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_target,
    predict_tomorrows_avg)


def page_forecast_body():
    """
    Page to display the Forecast Predictometer Interface

    Steps:
    1. Load the historical stock data and select key features for analysis

    2. Load the pre-trained models for:
    - Predicting tomorrow's average price (regression model)
    - Classifying if tomorrow's average price will be higher or
    lower (classification model)

    3. Display the client requirements for forecasting both the direction
    and the estimated price

    4. Optionally display the stock data for user inspection

    5. Generate a live data input interface to allow users to make predictions

    6. When "Run Predictive Analysis" is clicked:
    - Classify tomorrow's average price as higher or lower
    - Predict the estimated average price for tomorrow
    """
    df = load_stock_data(0)
    df = df[['close', 'open', 'pre_close', 'high', 'average']].copy()

    # load predict target files
    version = 'v1'
    target_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_target/{version}/clf_pipeline_model.pkl")
    target_features = (pd.read_csv(f"outputs/ml_pipeline/predict_target/"
                                   f"{version}/X_train.csv").columns.to_list())

    # load predict tomorrow's average files
    version = 'v1'
    tomorrows_avg_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_tomorrows_avg/"
        f"{version}/regressor_pipeline.pkl")
    tomorrows_avg_features = (pd.read_csv
                              (f"outputs/ml_pipeline/predict_tomorrows_avg/"
                               f"{version}/X_train.csv").columns.to_list())

    st.write("### Forecast Predictometer Interface")
    st.info(
        "* The client is interested in forecasting whether tomorrow's "
        "average price will be higher or lower compared to today's price. "
        "Additionally, the client wants to predict the expected price "
        "for tomorrow to gain insights into potential risk exposure. "
        "Based on this analysis, the likelihood of tomorrow's average "
        "price being higher or lower than today's price should be "
        "presented, along with the expected price estimate"
    )

    if st.checkbox("Stock Data"):
        st.write(
            f"* The dataset has {df.shape[0]} "
            f"rows and {df.shape[1]} columns")

        st.write(df)

    st.write("---")

    # Generate Live Data
    # check_variables_for_UI(target_features) tomorrows_avg_features
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        predict_target(X_live, target_features,
                       target_pipe_model)

        predict_tomorrows_avg(X_live, tomorrows_avg_features,
                              tomorrows_avg_pipe)


def check_variables_for_UI(target_features):
    """
    Displays the unique set of features used as inputs across all pipelines
    (classification, regression, clustering) for the user interface

    Args:
        target_features (iterable): Iterable of feature names used in pipelines

    Side Effects:
        Writes to the Streamlit UI the count and list of unique features
    """
    combined_features = set(
        list(
            itertools.chain(target_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n"
        f"{combined_features}")


def DrawInputsWidgets():
    """
    Diaplys interactive input widgets for key stock features
    to generate live data for prediction

    Steps:
    1. Load historical stock data and define a percentage
    range for widget values

    2. Create five columns for widget inputs: 'close', 'open',
    'pre_close', 'high', and 'average'

    3. For each feature, display a number input widget with
    dynamic min, max, and median values

    4. Calculate the average of 'open' and 'close' and display it as a metric

    5. Return the live DataFrame (`X_live`) with the collected inputs
    for real-time prediction
    """
    # load dataset
    df = load_stock_data(0)
    # Set the range as a percentage of the dataset's min and max values
    # Minimum is 40% of the lowest value in the dataset
    # Maximum is 200% (double) of the highest value in the dataset
    percentageMin, percentageMax = 0.4, 2.0

    # we create input widgets only for 4 features
    col1, col2, col3, col4, col5 = st.columns(5)

    # We are using these features to feed the ML pipeline - values
    # copied from check_variables_for_UI() result
    # {"close","open","pre_close","high"} {"average"}

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on
    # the variable type (numerical or categorical)
    # and set initial values
    with col1:
        feature = "close"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col2:
        feature = "open"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col3:
        feature = "pre_close"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col4:
        feature = "high"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    # Display the average of "open" and "close"
    # store it in X_live['average']
    with col5:
        avg_value = (X_live["open"].item() + X_live["close"].item()) / 2
        st.metric(label="average", value=f"{avg_value:.2f}")
        X_live["average"] = avg_value

    # st.write(X_live)

    return X_live
