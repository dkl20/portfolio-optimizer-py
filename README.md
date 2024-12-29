# portfolio-optimizer-py

This program applies the Modern Portfolio Theory approach to determine the optimal risk and reward allocations for your specified set of stocks and the given length of monthly historical data. It calculates historical average returns and standard deviations (risk), using these as assumptions for expected future returns. The program also uses a covariance matrix to calculate the relationship between the movements of each stock, measuring how they move together. It then calculates the most optimal combination of weightings for each stock by identifying the allocation that provides the highest return for the least amount of risk, in other words, by maximizing the Sharpe Ratio.

The program imports data from the Yahoo Finance API to compute the metrics for each stock. Additionally, it incorporates IRX, the 13-week Treasury bill yield (issued by the U.S.), as the risk-free asset.

A simplified quadratic Mean-Variance Utility Function is used to determine the optimal allocation between risky assets and the risk-free asset, based on the user's level of risk aversion.
