import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime, timedelta


def set_cache_expiration():
    """
    Sets the cache expiration time if it's not already in session state
    """
    if 'cache_expiration_time' not in st.session_state:
        st.session_state.cache_expiration_time = (
            datetime.now() + timedelta(seconds=1000))

    return st.session_state.cache_expiration_time


@st.cache_data(ttl=1000)
def load_live_stock_data(ticker_symbol):
    """
    - Cache for one day (86400 seconds) so data auto refreshes daily
    - Fetches the most recent 10 trading days for the given ticker
    - If it fails, return None and the error message as a string
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="10d", actions=False)
        df.index = df.index.tz_localize(None)
        df.columns = df.columns.str.lower()
        df['average'] = df[['open', 'close']].mean(axis=1)
        set_cache_expiration()

        return df, None

    except Exception as e:

        return None, str(e)


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
