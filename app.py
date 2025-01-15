import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Portfolio Optimization Application")

# Load the data directly from a fixed backend file
@st.cache_data
def load_data():
    file_path = "Returns_Data.xlsx"  # Path to your fixed backend file
    data = pd.read_excel(file_path, sheet_name=0, parse_dates=['Date'], index_col='Date')
    # Fill NaN values with 0 (if any)
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
    "SPGTIND Index": "S&P GLOBAL INFRASTRUCTUR",
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

# Calculate daily returns in percentage
def calculate_monthly_returns(data):
    monthly_returns = data.pct_change()
    monthly_returns.fillna(0, inplace = True)
    #print(monthly_returns)
    return monthly_returns

# Calculate cumulative annualized returns
def calculate_cumulative_annualized_returns(data):
    # Calculate annual returns
    annual_returns = data.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    
    # Calculate cumulative annualized returns
    cumulative_annualized_returns = (1 + annual_returns).cumprod() - 1
    
    # Replace any inf or -inf values with 0
    cumulative_annualized_returns.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Fill NaN values with 0 (if any)
    cumulative_annualized_returns.fillna(0, inplace=True)
    
    return cumulative_annualized_returns

# Function to calculate portfolio returns based on user-defined allocations
def calculate_portfolio_returns(data, allocations):
    portfolio_returns = data.mul(allocations, axis=1).sum(axis=1)
    portfolio_monthly_returns = portfolio_returns.pct_change().fillna(0)
    return portfolio_monthly_returns

# Function to calculate volatility
def calculate_yearly_volatility(data):
    # Resample to yearly data and calculate volatility
    yearly_volatility = data.resample('YE').std() * np.sqrt(12)
    return yearly_volatility

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
    # Calculate drawdowns
    cumulative_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns/cumulative_max) - 1
    return drawdowns

# Streamlit UI
st.title("Portfolio Optimization Application")

# Year range selection with a slider
min_year, max_year = data.index.year.min(), data.index.year.max()
year_range = st.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

# Filter data for the selected year range
filtered_data = filtered_data[(filtered_data.index.year >= year_range[0]) & (filtered_data.index.year <= year_range[1])]

# User-defined allocations
st.header("Portfolio % Allocation")

allocations = {}
initial_allocation = 100 / len(fixed_assets)
non_gold_assets = [asset for asset in fixed_assets if asset != "GOLDLNPM Index"]

for asset in non_gold_assets:
    allocations[asset] = initial_allocation

n_inputs_per_line = 6
total_alloc = 0
for i, asset in enumerate(non_gold_assets):
    col_idx = i % n_inputs_per_line
    if col_idx == 0:
        cols = st.columns(n_inputs_per_line)

    text_value = cols[col_idx].text_input(
        f"{asset_labels[asset]}",
        value=int(allocations[asset]),
        key=f"{asset}_input",
        max_chars=4
    )

    try:
        input_value = int(text_value)
        if input_value < 0 or input_value > 100:
            raise ValueError("Input must be between 0 and 100.")
    except ValueError:
        st.warning(f"Please enter a valid number between 0 and 100 for {asset_labels[asset]}.")
        input_value = allocations[asset]

    allocations[asset] = input_value
    total_alloc = sum(allocations[asset] for asset in non_gold_assets)

# Automatically calculate GOLDLNPM Index allocation
gold_alloc = max(0, 100 - total_alloc)
allocations["GOLDLNPM Index"] = gold_alloc

# Display GOLDLNPM Index allocation
st.write(f"### {asset_labels['GOLDLNPM Index']} Allocation: {gold_alloc}")

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
#st.subheader("Select Asset Classes to Display Graphs")
#selected_assets = []
#for asset in fixed_assets:
#    if st.checkbox(f"{asset_labels[asset]}", value=True):
#        selected_assets.append(asset)

# Checkbox for each asset class
st.subheader("Select Asset Classes to Display Graphs")
selected_assets = []

# Add a "Select All" checkbox
select_all = st.checkbox("Select All", value=True, key="select_all_checkbox")

# Define the number of checkboxes per row
checkboxes_per_row = 4  # Adjust based on desired layout

# Create rows of checkboxes
cols = st.columns(checkboxes_per_row)

for i, asset in enumerate(fixed_assets):
    col = cols[i % checkboxes_per_row]  # Distribute checkboxes across columns
    if col.checkbox(f"{asset_labels[asset]}", value=select_all, key=f"{asset}_checkbox"):
        selected_assets.append(asset)

# Button to generate/update graphs
# Button to generate/update graphs
if st.button("Generate/Update Graphs", disabled=generate_button_disabled):
    #weights = np.array([allocations[asset] / 100 for asset in fixed_assets])

    # Calculate daily and cumulative annualized returns
    monthly_returns = calculate_monthly_returns(filtered_data)

    # Calculate portfolio monthly returns
    portfolio_returns = calculate_portfolio_returns(filtered_data, allocations)
    portfolio_df = portfolio_returns.to_frame("Portfolio")
    
    # Calculate cumulative annualized returns for all assets
    cumulative_returns = calculate_cumulative_annualized_returns(monthly_returns)
    portfolio_cumulative_returns = calculate_cumulative_annualized_returns(portfolio_df)

    #st.write(portfolio_cumulative_returns)

    # Plot cumulative annualized returns for all selected assets and portfolio on a single graph
    st.subheader("Cumulative Annualized Returns")
    plt.figure(figsize=(14, 8))  # Adjust figure size for combined graph
    for asset in selected_assets:
        plt.plot(cumulative_returns.index.year, cumulative_returns[asset], label=asset_labels[asset])
    plt.plot(portfolio_cumulative_returns.index.year, portfolio_cumulative_returns["Portfolio"], label="Portfolio", color="black", linestyle="--")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Year")
    plt.ylabel("Cumulative Annualized Return (%)")
    plt.title("Cumulative Annualized Returns of Selected Assets and Portfolio")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
    plt.grid(True)
    st.pyplot(plt)

    # Calculate yearly volatility for the selected year range and selected assets
    yearly_volatility = calculate_yearly_volatility(monthly_returns)
    portfolio_yearly_volatility = calculate_yearly_volatility(portfolio_df)

    # Plot yearly volatility for all selected assets and portfolio
    st.subheader("Yearly Volatility")
    plt.figure(figsize=(14, 8))  # Adjust figure size for combined graph
    for asset in selected_assets:
        plt.plot(yearly_volatility.index.year, yearly_volatility[asset], label=asset_labels[asset])
    plt.plot(portfolio_yearly_volatility.index.year, portfolio_yearly_volatility["Portfolio"], label="Portfolio", linewidth=2, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Year")
    plt.ylabel("Volatility (%)")
    plt.title("Yearly Volatility of Selected Assets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
    plt.grid(True)
    st.pyplot(plt)

    # Extract T-bill data for Sharpe Ratio
    t_bill_data = monthly_returns["LD12TRUU Index"]
    
    # Calculate yearly Sharpe Ratio for the selected year range and selected assets
    yearly_sharpe_ratio = calculate_yearly_sharpe_ratio(monthly_returns, t_bill_data)
    portfolio_yearly_sharpe_ratio = calculate_yearly_sharpe_ratio(portfolio_df, t_bill_data)

    # Plot yearly Sharpe Ratio for all selected assets and portfolio
    st.subheader("Yearly Sharpe Ratio")
    plt.figure(figsize=(14, 8))  # Adjust figure size for combined graph
    for asset in selected_assets:
        plt.plot(yearly_sharpe_ratio.index.year, yearly_sharpe_ratio[asset], label=asset_labels[asset])
    plt.plot(portfolio_yearly_sharpe_ratio.index.year, portfolio_yearly_sharpe_ratio["Portfolio"], label="Portfolio", linewidth=2, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Year")
    plt.ylabel("Sharpe Ratio")
    plt.title("Yearly Sharpe Ratio of Selected Assets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
    plt.grid(True)
    st.pyplot(plt)

    # Calculate drawdowns
    asset_drawdowns = calculate_drawdowns(cumulative_returns)
    portfolio_drawdowns = calculate_drawdowns(portfolio_cumulative_returns)

    # Plot drawdowns for selected assets and portfolio
    st.subheader("Drawdowns")
    plt.figure(figsize=(14, 8))  # Adjust figure size for combined graph
    for asset in selected_assets:
        plt.plot(asset_drawdowns.index.year, asset_drawdowns[asset], label=asset_labels[asset])
    plt.plot(portfolio_drawdowns.index.year, portfolio_drawdowns["Portfolio"], label="Portfolio", linewidth=2, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Year")
    plt.ylabel("Drawdowns")
    plt.title("Drawdowns of Selected Assets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
    plt.grid(True)
    st.pyplot(plt)