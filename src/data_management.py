import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import time
from datetime import (
    datetime,
    timedelta,
    timezone)


@st.cache_data(ttl=3599, show_spinner=False)
def load_live_stock_data(ticker_symbol):
    """
    - Cache for one hour (3599 seconds) so data auto refreshes daily

    - Fetches the most recent 10 trading days for the given ticker,
    and returns a fixed expiration time

    - If it fails, return None, a fixed expiration time,
    and the error message as a string
    """
    with st.spinner('Loading Data...'):
        time.sleep(4)
        expiration_time = datetime.now(timezone.utc) + timedelta(seconds=3599)
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="10d", actions=False)
            df.index = df.index.tz_localize(None)
            df.columns = df.columns.str.lower()
            df['average'] = df[['open', 'close']].mean(axis=1)

            return df, expiration_time, None

        except Exception as e:

            return None, expiration_time, str(e)


@st.cache_data
def load_stock_data(n):
    """
    Loads stock data from CSV files based on the input parameter

    Args:
        n (int): If 0, loads the cleaned dataset; otherwise,
        loads the raw dataset

    Returns:
        pd.DataFrame: The loaded stock data
    """
    if n == 0:
        df = pd.read_csv("outputs/datasets/cleaned/phnx_2010_2025.csv")
    else:
        df = pd.read_csv("outputs/datasets/collection/phnx_2010_2025.csv")
    return df


def load_pkl_file(file_path):
    """
    Loads a pickled file using joblib

    Args:
        file_path (str): Path to the .pkl file

    Returns:
        object: The deserialized Python object stored in the file
    """
    return joblib.load(filename=file_path)
