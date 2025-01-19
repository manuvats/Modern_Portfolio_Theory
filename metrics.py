# Calculate monthly returns in percentage
def calculate_monthly_returns(data):
    monthly_returns = data.pct_change()
    #monthly_returns.fillna(0, inplace=True)
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