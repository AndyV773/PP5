import streamlit as st
import pandas as pd
from src.data_management import load_stock_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_target)


def page_prospect_body():

    # load predict target files
    version = 'v1'
    target_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_target/{version}'
        '/clf_pipeline_data_cleaning_feat_eng.pkl')
    target_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_target/{version}/clf_pipeline_model.pkl")
    target_features = (pd.read_csv(f"outputs/ml_pipeline/predict_target/"
                                   f"{version}/X_train.csv").columns.to_list())

    st.write("### Prospect Predictometer Interface")
    st.info(
        "* The client is interested in determining whether tomorrow's "
        "average price will be higher or lower compared to today's price. "
        "Additionally, the client wants to understand the expected price "
        "range for tomorrow to gain insights into potential risk exposure. "
        "Based on this analysis, key factors that could sustain or enhance "
        "price stability, as well as signal high-risk volatility, "
        "should be identified and presented"
    )
    st.write("---")

    # Generate Live Data
    # check_variables_for_UI(tenure_features, churn_features, cluster_features)
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        predict_target(X_live, target_features,
                       target_pipe_dc_fe, target_pipe_model)


def check_variables_for_UI(target_features):
    import itertools

    # The widgets inputs are the features used in all
    # pipelines (tenure, churn, cluster)
    # We combine them only with unique values
    combined_features = set(
        list(
            itertools.chain(target_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n"
        f"{combined_features}")


def DrawInputsWidgets():

    # load dataset
    df = load_stock_data(0)
    percentageMin, percentageMax = 0.4, 2.0

# we create input widgets only for 6 features
    col1, col2, col3 = st.columns(3)

    # We are using these features to feed the ML pipeline - values
    # copied from check_variables_for_UI() result
    # {"close","open","pre_close"}

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

    # st.write(X_live)

    return X_live
