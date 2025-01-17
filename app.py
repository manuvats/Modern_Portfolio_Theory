import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Portfolio Optimization Application")

# Load the data directly from a fixed backend file
@st.cache_data
def load_data():
    file_path = "Returns_Data.xlsx"  # Path to your fixed backend file
    data = pd.read_excel(file_path, sheet_name=0, parse_dates=['Date'], index_col='Date')
    data.fillna(0, inplace=True)
    return data

# Load the data
data = load_data()

# Define fixed asset classes
asset_labels = {
    "SPX Index": "S&P 500 INDEX",
    "RTY Index": "RUSSELL 2000 INDEX",
    "MXEA Index": "MSCI EAFE",
    "MXEF Index": "MSCI EM",
    "FNER Index": "FTSE NAREIT All Eq REITS",
    "BCOMTR Index": "BBG Commodity TR",
    "GOLDLNPM Index": "LBMA Gold Price PM USD",
    "SPGTIND Index": "S&P GLOBAL INFRASTRUCTURE",
    "LD12TRUU Index": "Bloomberg US Treasury Bill",
    "LBUTTRUU Index": "Bloomberg US Treasury Inflation-Linked Bond Index",
    "LBUSTRUU Index": "U.S. Aggregate",
    "LF98TRUU Index": "Bloomberg U.S. Corporate High Yield",
    "LMBITR Index": "Bloomberg Municipal Bond Index",
    "BTSYTRUH Index": "Bloomberg Global Treasury Index, USD Hedged",
    "LEGATRUH Index": "Bloomberg GlobalAgg Index, USD Hedged",
    "BSSUTRUU Index": "Bloomberg EM USD Aggregate: Sovereign Index",
    "EMUSTRUU Index": "Bloomberg Emerging Markets Hard Currency Aggregate Index",
    "HEDGNAV Index": "Credit Suisse Hedge Fund Index",
    "HFRIFWI Index": "HFRI Fund Weighted Composite Index"
}

fixed_assets = list(asset_labels.keys())

# Filter data to include only the fixed asset classes
filtered_data = data[fixed_assets]

# Calculate monthly returns in percentage
def calculate_monthly_returns(data):
    monthly_returns = data.pct_change()
    monthly_returns.fillna(0, inplace=True)
    return monthly_returns

# Function to calculate portfolio returns based on user-defined allocations
def calculate_portfolio_returns(data, allocations):
    portfolio_returns = data.mul(allocations, axis=1).sum(axis=1)
    portfolio_monthly_returns = portfolio_returns.pct_change().fillna(0)
    return portfolio_monthly_returns

def calculate_cumulative_returns(data):
    cumulative_returns = (1 + data).cumprod() - 1
    return cumulative_returns

#def calculate_volatility(data):
#    return data.std() * np.sqrt(12)

# Function to calculate yearly volatility
def calculate_yearly_volatility(data):
    # Resample to yearly data and calculate volatility
    yearly_volatility = data.resample('YE').std() * np.sqrt(12)
    return yearly_volatility

#def calculate_yearly_sharpe_ratio(data, risk_free_rate):

    # Align the risk-free rate with the data index
    #risk_free_rate = risk_free_rate.reindex(data.index, method='ffill')
    #excess_returns = data.sub(risk_free_rate, axis=0)
    #sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    #return sharpe_ratio

# Function to calculate yearly Sharpe Ratio
def calculate_yearly_sharpe_ratio(data, risk_free_rate):

    # Align the risk-free rate with the data index
    risk_free_rate = risk_free_rate.reindex(data.index, method='ffill')

    # Calculate excess returns
    excess_returns = data.sub(risk_free_rate, axis=0)

    # Calculate yearly average returns and standard deviation
    yearly_avg_returns = excess_returns.resample('YE').mean()
    yearly_std_dev = excess_returns.resample('YE').std()

    # Compute Sharpe Ratio
    yearly_sharpe_ratio = (yearly_avg_returns / yearly_std_dev) * np.sqrt(12)
    #print(risk_free_rate)

    return yearly_sharpe_ratio

def calculate_drawdowns(cumulative_returns):
    cumulative_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / cumulative_max) - 1
    return drawdowns

# Streamlit UI
st.title("Portfolio Optimization Application")

# Convert pandas.Timestamp to datetime.date
min_date = data.index.min().date()
max_date = data.index.max().date()

# Use datetime.date in the slider
date_range = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

def filter_data_by_date(df, start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df.loc[mask]

# Filter data for the selected date range
filtered_data = filter_data_by_date(filtered_data, date_range[0], date_range[1])

# User-defined allocations
st.header("Portfolio % Allocation")

allocations = {}
initial_allocation = 100 / len(fixed_assets)
non_gold_assets = [asset for asset in fixed_assets if asset != "GOLDLNPM Index"]

cols = st.columns(6)
total_alloc = 0
for i, asset in enumerate(non_gold_assets):
    col = cols[i % 6]
    input_value = col.number_input(
        f"{asset_labels[asset]}",
        min_value=0,
        max_value=100,
        value=int(initial_allocation),
        step=1,
        key=f"{asset}_input"
    )
    allocations[asset] = input_value
    total_alloc += input_value

# Automatically calculate GOLDLNPM Index allocation
gold_alloc = max(0, 100 - total_alloc)
allocations["GOLDLNPM Index"] = gold_alloc

# Display GOLDLNPM Index allocation
st.write(f"### {asset_labels['GOLDLNPM Index']} Allocation: {gold_alloc}%")

# Display total allocation prominently
if total_alloc > 100:
    st.error("Total allocation exceeds 100%. Please correct the inputs.")
    generate_button_disabled = True
else:
    generate_button_disabled = False
    st.markdown(
        f"<h2 style='text-align: center;'>Total Allocation: {total_alloc + gold_alloc}%</h2>",
        unsafe_allow_html=True
    )

# Checkbox for each asset class
st.subheader("Select Asset Classes to Display Graphs")
selected_assets = []
select_all = st.checkbox("Select All", value=True, key="select_all_checkbox")

cols = st.columns(4)
for i, asset in enumerate(fixed_assets):
    col = cols[i % 4]
    if col.checkbox(f"{asset_labels[asset]}", value=select_all, key=f"{asset}_checkbox"):
        selected_assets.append(asset)

if st.button("Generate/Update Graphs", disabled=generate_button_disabled):
    monthly_returns = calculate_monthly_returns(filtered_data)
    cumulative_returns = calculate_cumulative_returns(monthly_returns)

    # Calculate portfolio monthly returns
    portfolio_returns = calculate_portfolio_returns(filtered_data, allocations)
    portfolio_df = portfolio_returns.to_frame("Portfolio")

    #portfolio_returns = pd.DataFrame({"Portfolio": np.dot(monthly_returns[selected_assets], [allocations[a] / 100 for a in selected_assets])})
    cumulative_portfolio_returns = calculate_cumulative_returns(portfolio_df)

    st.subheader("Cumulative Returns")
    plt.figure(figsize=(14, 8))
    for asset in selected_assets:
        plt.plot(cumulative_returns.index, cumulative_returns[asset], label=asset_labels[asset])
    plt.plot(cumulative_portfolio_returns.index, cumulative_portfolio_returns["Portfolio"], label="Portfolio", color="black", linestyle="--")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.title("Cumulative Returns of Selected Assets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    st.pyplot(plt)

    # Volatility
    #volatility = calculate_yearly_volatility(monthly_returns)
    #st.write(cumulative_returns.index)
    #st.write(volatility)
    #st.write(volatility[selected_assets])
    #asset_labels[selected_assets]
    
    # Calculate yearly volatility for the selected year range and selected assets
    yearly_volatility = calculate_yearly_volatility(monthly_returns)
    portfolio_yearly_volatility = calculate_yearly_volatility(portfolio_df)

    st.subheader("Yearly Volatility")
    plt.figure(figsize=(14, 8))
    for asset in selected_assets:
        plt.plot(yearly_volatility.index, yearly_volatility[asset], label=asset_labels[asset])
    plt.plot(portfolio_yearly_volatility.index.year, portfolio_yearly_volatility["Portfolio"], label="Portfolio", linewidth=2, linestyle='--')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.title("Yearly Volatility of Selected Assets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    st.pyplot(plt)

    # Sharpe Ratio
    #risk_free_rate = monthly_returns["LD12TRUU Index"]  # Example risk-free rate
    #sharpe_ratio = calculate_yearly_sharpe_ratio(monthly_returns, risk_free_rate)
    
    # Extract T-bill data for Sharpe Ratio
    t_bill_data = monthly_returns["LD12TRUU Index"]

    # Calculate yearly Sharpe Ratio for the selected year range and selected assets
    yearly_sharpe_ratio = calculate_yearly_sharpe_ratio(monthly_returns, t_bill_data)
    portfolio_yearly_sharpe_ratio = calculate_yearly_sharpe_ratio(portfolio_df, t_bill_data)

    st.subheader("Yearly Sharpe Ratio")
    plt.figure(figsize=(14, 8))
    for asset in selected_assets:
        plt.plot(yearly_sharpe_ratio.index, yearly_sharpe_ratio[asset], label=asset_labels[asset])
    plt.plot(portfolio_yearly_sharpe_ratio.index.year, portfolio_yearly_sharpe_ratio["Portfolio"], label="Portfolio", linewidth=2, linestyle='--')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.title("Yearly Sharpe Ratio of Selected Assets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    st.pyplot(plt)

    # Drawdowns
    #drawdowns = calculate_drawdowns(cumulative_returns)
    # Calculate drawdowns
    asset_drawdowns = calculate_drawdowns(cumulative_returns)
    portfolio_drawdowns = calculate_drawdowns(cumulative_portfolio_returns)

    st.subheader("Drawdowns")
    plt.figure(figsize=(14, 8))
    for asset in selected_assets:
        plt.plot(asset_drawdowns.index, asset_drawdowns[asset], label=asset_labels[asset])
    plt.plot(portfolio_drawdowns.index.year, portfolio_drawdowns["Portfolio"], label="Portfolio", linewidth=2, linestyle='--')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.title("Drawdowns of Selected Assets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    st.pyplot(plt)
