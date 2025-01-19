import streamlit as st
import pandas as pd

# Load the data directly from a fixed backend file
@st.cache_data
def load_data():
    file_path = "Returns_Data.xlsx"  # Path to your fixed backend file
    data = pd.read_excel(file_path, sheet_name=0, parse_dates=['Date'], index_col='Date')
    #data.fillna(0, inplace=True)
    return data

def filter_data_by_date(df, start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df.loc[mask]

def get_asset_date_ranges(data):
    date_ranges = {}
    for column in data.columns:
        valid_data = data[column].dropna()
        if not valid_data.empty:
            date_ranges[column] = {
                'start': valid_data.index.min().date(),
                'end': valid_data.index.max().date()
            }
    return date_ranges

