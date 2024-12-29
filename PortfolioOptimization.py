import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import scipy.optimize as sco

print("Please follow the following set instructions to obtain average returns of your specified stocks and standard deviations over your specified time period")

# ticker validation check
def get_valid_tickers():
    while True:
        tickers_input = input("Enter stock tickers (use a comma and a space to separate): ")
        tickers = tickers_input.split(', ')
        
        # Convert tickers to uppercase
        tickers = [ticker.upper() for ticker in tickers]
        
        # check validity of each ticker
        valid_tickers = []
        invalid_tickers = []
        for ticker in tickers:
            try:
                stock_info = yf.Ticker(ticker).history(period="1d")
                if not stock_info.empty:  # valid if the returned DataFrame is not empty
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except:
                # if there is any other errors like network error
                invalid_tickers.append(ticker)
        
        # If all tickers are valid, return them
        if not invalid_tickers:
            return valid_tickers
        
        # If there are invalid tickers, notify user and ask for re-entry
        print(f"The following tickers are invalid: {invalid_tickers}")
        print("Please re-enter your tickers.")

# obtain valid tickers from the user
tickers = get_valid_tickers()

# obtaining user input for time range calculations
def user_input_for_months():
    while True:
        try:
            months = int(input("How many months of data would you like to use for the calculations? "))
            if months <= 0:
                print("Please enter a positive number.")
            else:
                return months
        except ValueError:
            print("Invalid input. Please enter an integer.")
            
# ask user how many months of data to use
months = user_input_for_months()

# calculate the start date based on the current date minus the number of months
end_date = datetime.now()
start_date = end_date - timedelta(days=months * 30) # approximate month length of 30 days

# convert start_date to string format YYYY-MM-DD
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

print(f"Tickers: {tickers}")
print(f"Data Start date: {start_date_str}, End date: {end_date_str}")

# download historical data based on user input months
data = yf.download(tickers, start=start_date_str, end=end_date_str)['Adj Close'] 
print(data)

# resample data to monthly frequency ('ME') and calculate the monthly returns
monthly_data = data.resample('ME').last()  # Resample to the last trading day of each month

# calculate monthly returns: (price_end / price_start) - 1
monthly_returns = monthly_data.pct_change().dropna()  # pct_change() gives percentage change between periods (monthly)

# calculate average monthly return, standard deviation, and covariance for each stock
average_monthly_returns = monthly_returns.mean()
standard_deviation = monthly_returns.std()
covariance = monthly_returns.cov()

# risk-free rate (3-month T-bill IRX)
riskfreerate_info = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100  # Convert to decimal
print(f"Risk-Free Rate (3-month T-Bill): {riskfreerate_info:.4f}")

# calculate excess returns (stock returns above the risk-free rate)
excess_returns = average_monthly_returns - riskfreerate_info

# Print the average monthly returns, standard deviation, and covariance matrix for each stock
print("Average monthly returns for each stock (over the specified period):")
for ticker, avg_return in average_monthly_returns.items():
    print(f"{ticker}: {avg_return:.4f}")

print("\nStandard Deviation (Risk) of monthly returns for each stock:")
for ticker, risk in standard_deviation.items():
    print(f"{ticker}: {risk:.4f}")

print("\nCovariance matrix of monthly returns:")
print(covariance)

# function to calculate portfolio return (using excess returns)
def portfolio_expected_return(weights, excess_returns):
    return np.dot(weights, excess_returns)

# function to calculate portfolio volatility (risk)
def portfolio_risk(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

# function to calculate the Sharpe Ratio (using excess returns)
def sharpe_ratio(weights, excess_returns, covariance_matrix):
    portfolio_return = portfolio_expected_return(weights, excess_returns)
    portfolio_volatility = portfolio_risk(weights, covariance_matrix)
    return portfolio_return / portfolio_volatility  # maximize this

# function to minimize the negative Sharpe Ratio (for optimization)
def minimize_sharpe_ratio(weights, excess_returns, covariance_matrix):
    return -sharpe_ratio(weights, excess_returns, covariance_matrix)

# function to calculate the optimal portfolio weights
def optimal_portfolio_weights(excess_returns, covariance_matrix):
    num_assets = len(excess_returns)
    
    # Initial guess for the weights (equally distributed)
    initial_weights = np.ones(num_assets) / num_assets
    
    # Constraints: weights should sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # each weight must be between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Maximizing the Sharpe Ratio
    optimal_result = sco.minimize(minimize_sharpe_ratio, initial_weights, args=(excess_returns, covariance_matrix),
        method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_result.x  # Optimal weights

# get optimal weights for the portfolio
optimal_weights = optimal_portfolio_weights(excess_returns, covariance)

# calculate the expected return and risk for the optimized portfolio
optimized_return = portfolio_expected_return(optimal_weights, excess_returns) + riskfreerate_info  # add the risk-free rate to the expected return
optimized_risk = portfolio_risk(optimal_weights, covariance)

print("\nOptimal Portfolio Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

print(f"\nOptimized Portfolio Expected Return: {optimized_return:.4f}")
print(f"Optimized Portfolio Risk (Standard Deviation): {optimized_risk:.4f}")

# function to calculate utility
def utility(expected_return, risk, risk_aversion):
    return expected_return - 0.5 * risk_aversion * (risk ** 2)

# ask user for risk aversion level (A)
def get_risk_aversion():
    while True:
        try:
            risk_aversion = float(input("Enter your risk aversion level (e.g., 1 for low, 5 for high): "))
            if risk_aversion <= 0:
                print("Please enter a positive value for risk aversion.")
            else:
                return risk_aversion
        except ValueError:
            print("Invalid input. Please enter a number.")

risk_aversion = get_risk_aversion()

# optimal allocation between risky portfolio and risk-free asset
def optimal_risky_allocation(risk_aversion, optimized_return, optimized_risk, risk_free_rate):
    risky_weight = (optimized_return - risk_free_rate) / (risk_aversion * (optimized_risk ** 2))
    risky_weight = max(0, min(1, risky_weight))  
    return risky_weight

# calculate the allocation
risky_weight = optimal_risky_allocation(risk_aversion, optimized_return, optimized_risk, riskfreerate_info)
risk_free_weight = 1 - risky_weight

# portfolio utility
portfolio_utility = utility(
    expected_return=risky_weight * optimized_return + risk_free_weight * riskfreerate_info,
    risk=risky_weight * optimized_risk,
    risk_aversion=risk_aversion
)

print(f"\nOptimal Allocation:")
print(f"Risky Portfolio Weight: {risky_weight:.4f}")
print(f"Risk-Free Asset Weight: {risk_free_weight:.4f}")
print(f"Portfolio Utility: {portfolio_utility:.4f}")
