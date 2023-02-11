import numpy as np
import pandas as pd

def rolling_sharpe(daily_returns: pd.Series,
                   risk_free_rate: pd.Series | float, window: int) -> pd.Series:
    """
    Computes the rolling Sharpe ratio for a given ticker and risk-free rate.
    
    Parameters:
        daily_returns (pd.Series):
        risk_free_rate (pd.Series | float):
        window (int):

    Returns:
        pd.Series:  
    """

    DAYS_IN_MONTH = 21
    
    # Compute D-hat
    avg_return = daily_returns.rolling(window = window * DAYS_IN_MONTH).mean()
    periodized_avg_return = avg_return * window * DAYS_IN_MONTH
    avg_diff_return = periodized_avg_return - risk_free_rate

    # Compute sigma
    volatility = daily_returns.rolling(window = window * DAYS_IN_MONTH).std()
    periodized_volatility = volatility * np.sqrt(window)
    
    # Compute rolling Sharpe ratio: D-hat / sigma
    rolling_sharpe_ratio = (avg_diff_return / periodized_volatility).dropna()
    
    return rolling_sharpe_ratio