# Volatility
    #volatility = calculate_yearly_volatility(monthly_returns)
    #st.write(cumulative_returns.index)
    #st.write(volatility)
    #st.write(volatility[selected_assets])
    #asset_labels[selected_assets]
    
    """Calculate yearly volatility for the selected year range and selected assets
    
    #yearly_volatility = calculate_yearly_volatility(monthly_returns)
    #portfolio_yearly_volatility = calculate_yearly_volatility(portfolio_df)

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
"""
