import streamlit as st


def predict_target(X_live, target_features,
                   target_pipeline_model):

    # from live data, subset features related to this pipeline
    X_live_target = X_live.filter(target_features)

    # apply data cleaning / feat engine pipeline to live data
    X_live_target_dc_fe = X_live_target

    # predict
    target_prediction = target_pipeline_model.predict(X_live_target_dc_fe)
    target_prediction_proba = target_pipeline_model.predict_proba(
        X_live_target_dc_fe)
    # st.write(target_prediction_proba)

    # Create a logic to display the results
    target_prob = target_prediction_proba[0, target_prediction][0]*100
    if target_prediction == 1:
        target_result = 'higher'
    else:
        target_result = 'lower'

    statement = (
        f"### There is {target_prob.round(1)}% probability "
        f"that tomorrow's average forecast will be **{target_result}**")

    st.write(statement)

    return target_prediction


def predict_tomorrows_avg(X_live, tomorrows_avg_features,
                          tomorrows_avg_pipeline):

    # Subset features related to this pipeline
    X_live_tomorrows_avg = X_live.filter(tomorrows_avg_features)

    # Predict tomorrow's average price
    tomorrows_avg_prediction = tomorrows_avg_pipeline.predict(
        X_live_tomorrows_avg)

    # Since it's regression, we just display the raw prediction
    predicted_price = tomorrows_avg_prediction[0]

    # Construct the output statement
    statement = (
        f"The model predicts that tomorrow's average price "
        f"will be approximately **{predicted_price:.2f}**"
    )

    st.write(statement)
