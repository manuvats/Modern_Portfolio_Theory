import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import date
from helper import *
from metrics import *

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Portfolio Optimization Application")

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

asset_date_ranges = get_asset_date_ranges(data)

# Streamlit UI
st.title("Portfolio Optimization Application")

# Convert pandas.Timestamp to datetime.date
min_date = data.index.min().date()
max_date = data.index.max().date()

# Date range selection with slider
st.write("### Select Date Range")

# Place both date pickers in the same row
col1, col2 = st.columns([1, 1])  # Adjust the proportions to control the width of the columns

# Use datetime.date in the slider
#date_range_slider = st.slider(
#    "Select Date Range",
#    min_value=min_date,
#    max_value=max_date,
#    value=(min_date, max_date),
#    format="YYYY-MM-DD",
#    key="slider_date_range"
#)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date_picker"
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date_picker"
    )

if start_date > end_date:
    st.error("Start date must be before or equal to the end date.")
else:
    # Filter data for the selected date range
    filtered_data = filter_data_by_date(data, start_date, end_date)

non_gold_assets = [asset for asset in fixed_assets if asset != "GOLDLNPM Index"]

# User-defined allocations
st.header("Portfolio % Allocation")

# After loading data and getting date ranges, before allocation section:
def get_available_assets(asset_date_ranges, start_date, end_date):
    available_assets = []
    for asset in fixed_assets:
        if asset in asset_date_ranges:
            asset_dates = asset_date_ranges[asset]
            if start_date >= asset_dates['start'] and end_date <= asset_dates['end']:
                available_assets.append(asset)
    #st.write("Available assets:", available_assets)
    return available_assets

# Add debugging print statements

#st.write("Asset date ranges:", asset_date_ranges)
#st.write("Gold date range:", asset_date_ranges.get("XAU Curncy", None))

allocations = {}
cols = st.columns(4)
total_alloc = 0.0

# Set default allocation to distribute 100% among non-gold assets
# Calculate initial allocation
available_assets = get_available_assets(asset_date_ranges, start_date, end_date)
initial_allocation = 100.0 / (len(available_assets) - 1) if available_assets else 0.0
#-1 to exclude Gold in the initial allocation

for i, asset in enumerate(non_gold_assets):
    col = cols[i % 4]

    # Check if asset has data in selected date range
    asset_in_range = True
    date_range_text = ""
    if asset in asset_date_ranges:
        asset_dates = asset_date_ranges[asset]
        date_range_text = f"({asset_dates['start']} to {asset_dates['end']})"
        asset_in_range = (
            start_date >= asset_dates['start'] and 
            end_date <= asset_dates['end']
        )
    
    if asset_in_range:
        input_value = col.number_input(
            f"{asset_labels[asset]}\n{date_range_text}",
            min_value=0.0,
            max_value=100.0,
            value=initial_allocation,
            step=0.01,
            key=f"{asset}_input"
        )
        allocations[asset] = input_value
        total_alloc += input_value
    else:
        col.text_input(
            f"{asset_labels[asset]}\n{date_range_text}",
            value="Data not available",
            disabled=True,
            key=f"{asset}_input_disabled"
        )
        allocations[asset] = 0.0

    #total_alloc += input_value
    
total_alloc = round(total_alloc, 2)

# Automatically calculate GOLDLNPM Index allocation
gold_alloc = round(max(0.0, 100.0 - total_alloc), 2)
allocations["GOLDLNPM Index"] = gold_alloc

# Display GOLDLNPM Index allocation
st.write(f"### {asset_labels['GOLDLNPM Index']} Allocation: {gold_alloc}%")

# Display total allocation prominently
if total_alloc > 100.0:
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
#select_all = st.checkbox("Select All", value=True, key="select_all_checkbox")

# Modify the checkbox section:
#selected_assets = {}
#cols = st.columns(6)
#for i, asset in enumerate(non_gold_assets):
#    col = cols[i % 6]
#    asset_in_range = asset in available_assets
    
#    if asset_in_range:
#        selected_assets[asset] = col.checkbox(
#            f"Select {asset_labels[asset]}",
#            #value=True,
#            key=f"alloc_{asset}_checkbox"
#        )
#    else:
#        selected_assets[asset] = col.checkbox(
#            f"Select {asset_labels[asset]}",
#            #value=False,
#            disabled=True,
#            key=f"alloc_{asset}_checkbox"
#        )

# Section for graph selection
st.subheader("Select Assets to Display in Graphs")
select_all = st.checkbox("Select All", value=True, key="graph_select_all")
selected_graph_assets = []

cols = st.columns(4)

for i, asset in enumerate(fixed_assets):
    col = cols[i % 4]
    
    asset_in_range = asset in available_assets

    if asset_in_range:
        if col.checkbox(
            f"{asset_labels[asset]}", 
            value=select_all, 
            key=f"graph_{asset}_checkbox"
            ):
            selected_graph_assets.append(asset)
    else:
        col.checkbox(
            f"{asset_labels[asset]}", 
            value=False,
            disabled=True,
            key=f"graph_{asset}_checkbox"
        )

if st.button("Generate/Update Graphs", disabled=generate_button_disabled):
    # Calculate returns only for available and selected assets
    available_selected_assets = [asset for asset in selected_graph_assets if asset in available_assets]
    
    if available_selected_assets:
        monthly_returns = calculate_monthly_returns(filtered_data[available_selected_assets])
        cumulative_returns = calculate_cumulative_returns(monthly_returns)
        

    # Calculate portfolio returns using all allocation assets
    portfolio_returns = calculate_portfolio_returns(filtered_data, allocations)
    portfolio_df = portfolio_returns.to_frame("Portfolio")

    #portfolio_returns = pd.DataFrame({"Portfolio": np.dot(monthly_returns[selected_assets], [allocations[a] / 100 for a in selected_assets])})
    cumulative_portfolio_returns = calculate_cumulative_returns(portfolio_df)

    st.subheader("Cumulative Returns")
    # Create figure
    fig = go.Figure()

    # Base trace settings
    base_trace = dict(
        mode='lines',
        hovertemplate='<b>%{customdata}</b><br>Date: %{x}<br>Return: %{y:.2f}x<br><extra></extra>'
    )

    # Add asset traces
    for asset in available_selected_assets:
        if asset in cumulative_returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[asset],
                    name=asset_labels[asset],
                    customdata=[asset_labels[asset]]*len(cumulative_returns),
                    line=dict(width=2),
                    **base_trace
                )
            )

    # Add portfolio trace
    fig.add_trace(
        go.Scatter(
            x=cumulative_portfolio_returns.index,
            y=cumulative_portfolio_returns["Portfolio"],
            name="Portfolio",
            line=dict(color='black', dash='dash', width=2),
            **base_trace
        )
    )

    # Layout configuration
    fig.update_layout(
        height=900,
        hovermode='closest',
        hoverdistance=100,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.4,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            itemsizing='constant'
        ),
        xaxis=dict(
            type='date',
            tickformat='%b %Y',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            tickangle=45,
            autorange=True,
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            tickformat='.2f',
            gridcolor='lightgrey',
            title="Cumulative Return (times)",
            autorange=True
        ),
        plot_bgcolor='white',
        margin=dict(b=100)
    )

    # Display chart with config
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape']
    })
    