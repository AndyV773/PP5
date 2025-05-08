import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_stock_data(n):
    if n == 0:
        df = pd.read_csv("outputs/datasets/cleaned/phnx_2010_2025.csv")
    else:
        df = pd.read_csv("outputs/datasets/collection/phnx_2010_2025.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)
