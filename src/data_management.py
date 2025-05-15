import streamlit as st
import pandas as pd
import joblib


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
