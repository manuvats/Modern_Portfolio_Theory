import numpy as np
import pandas as pd
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
def calculate_volatility(data):
    return data.std() * np.sqrt(12)

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(data, risk_free_rate):
    """Calculate Sharpe ratio"""
    # Ensure risk-free rate is aligned with data
    risk_free_rate = risk_free_rate.reindex(data.index, method='ffill')
    
    # Calculate excess returns
    excess_returns = data.sub(risk_free_rate, axis=0)
    
    # Calculate annualized Sharpe ratio
    return np.sqrt(12) * (excess_returns.mean() / excess_returns.std())

def calculate_drawdowns(cumulative_returns):
    cumulative_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / cumulative_max) - 1
    return drawdowns

'''def calculate_max_drawdown_info(cumulative_returns):
    """Calculate maximum drawdown and its timing"""
    drawdowns = calculate_drawdowns(cumulative_returns)
    max_drawdown = drawdowns.min()
    max_drawdown_idx = drawdowns.idxmin()
    
    max_drawdown_info = {
        'max_drawdown': max_drawdown,
        'date': max_drawdown_idx,
        'year': max_drawdown_idx.year,
        'month': max_drawdown_idx.month
    }
    
    return max_drawdown_info'''

def calculate_max_drawdown_info(returns):
    """Calculate maximum drawdown and its timing"""
    drawdowns = calculate_drawdowns(returns)
    
    if isinstance(returns, pd.Series):
        max_drawdown = drawdowns.min()
        max_drawdown_idx = drawdowns.idxmin()
        
        max_drawdown_info = {
            'max_drawdown': max_drawdown,
            'date': max_drawdown_idx,
            'year': max_drawdown_idx.year if hasattr(max_drawdown_idx, 'year') else None,
            'month': max_drawdown_idx.month if hasattr(max_drawdown_idx, 'month') else None
        }
    else:
        max_drawdown = drawdowns.min()
        max_drawdown_info = {
            'max_drawdown': max_drawdown,
            'date': None,
            'year': None,
            'month': None
        }
    
    return max_drawdown_info