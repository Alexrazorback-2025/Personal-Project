import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# ---- USER INPUT ----
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2018-01-01"
end_date = "2023-12-31"

# ---- DOWNLOAD AND CLEAN DATA ----
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)
    if len(tickers) == 1:
        return data['Adj Close'].to_frame(name=tickers[0])
    else:
        return data.loc[:, (slice(None), 'Adj Close')].droplevel(1, axis=1)

data = get_data(tickers, start_date, end_date)
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
std_devs = returns.std()

# ---- PORTFOLIO OPTIMIZATION ----
def portfolio_metrics(weights, mean_returns, cov_matrix):
    port_return = np.sum(weights * mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_vol
    return port_return, port_vol, sharpe

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = [1. / num_assets] * num_assets
    result = minimize(lambda w: -portfolio_metrics(w, *args)[2],
                      initial, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

result = optimize_portfolio(mean_returns, cov_matrix)
weights = result.x
port_return, port_vol, sharpe = portfolio_metrics(weights, mean_returns, cov_matrix)

# ---- OUTPUT RESULTS ----
print("\n--- Portfolio Metrics ---")
print(f"Expected Annual Return: {port_return:.2%}")
print(f"Volatility: {port_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

print("\n--- Optimized Weights ---")
for t, w in zip(tickers, weights):
    print(f"{t}: {w:.2%}")

# ---- VISUALIZATIONS ----

# 1. Pie Chart of Portfolio Allocation
def plot_allocation_pie(weights, tickers):
    fig, ax = plt.subplots()
    ax.pie(weights, labels=tickers, autopct="%1.1f%%", startangle=90)
    ax.set_title("Optimized Portfolio Allocation (Pie Chart)")
    plt.show()

# 2. Bar Chart of Portfolio Allocation
def plot_allocation_bar(weights, tickers):
    fig, ax = plt.subplots()
    ax.bar(tickers, weights, color='skyblue')
    ax.set_title("Optimized Portfolio Allocation (Bar Chart)")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    plt.grid(True)
    plt.show()

# 3. Cumulative Returns

def plot_cumulative_returns(returns):
    cum_returns = (1 + returns).cumprod()
    cum_returns.plot(figsize=(10, 5))
    plt.title("Cumulative Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1 Investment")
    plt.grid(True)
    plt.show()

# 4. Risk vs Return Scatter Plot
def plot_risk_return_scatter(mean_returns, std_devs, tickers):
    fig, ax = plt.subplots()
    ax.scatter(std_devs, mean_returns, s=100, c='green', alpha=0.7)
    for i, txt in enumerate(tickers):
        ax.annotate(txt, (std_devs[i], mean_returns[i]), fontsize=10, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel("Risk (Standard Deviation)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Risk vs. Return for Individual Assets")
    plt.grid(True)
    plt.show()

# 5. Correlation Heatmap
def plot_correlation_heatmap(data):
    correlation = data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Adjusted Close Prices")
    plt.tight_layout()
    plt.show()

# ---- CALL PLOTS ----
plot_allocation_pie(weights, tickers)
plot_allocation_bar(weights, tickers)
plot_cumulative_returns(returns)
plot_risk_return_scatter(mean_returns, std_devs, tickers)
plot_correlation_heatmap(data)# Personal-Project
