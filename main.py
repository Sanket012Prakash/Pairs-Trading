def main():

    # List of stock tickers for the pairs trading strategy
    # These tickers are primarily from the US market and include major tech, finance, energy, and manufacturing companies.
    symbols = [
        "MSFT", "AAPL", "GOOG", "AMZN", "META",
        "BRK-B", "JPM", "V", "JNJ", "PG",
        "XOM", "CVX", "BHP", "RIO", "TSLA",
        "NVDA", "AMD", "INTC", "NFLX", "ADBE",
        "CRM", "ORCL", "CSCO"
    ]


    training_start_date = '2019-04-01'
    training_end_date = '2022-03-31'
    testing_start_date = '2022-04-01'
    testing_end_date = '2025-03-31'


    pairs_trading = PairsTrading(
        symbols=symbols,
        start_date=training_start_date,
        end_date=testing_end_date,
        initial_capital=100000
    )


    pairs_trading.load_data()


    pairs_trading.find_cointegrated_pairs(
        training_start_date=training_start_date,
        training_end_date=training_end_date
    )


    pairs_trading.select_best_pairs(max_pairs=3)


    pairs_trading.backtest_portfolio(
        training_start_date=training_start_date,
        training_end_date=training_end_date,
        testing_start_date=testing_start_date,
        testing_end_date=testing_end_date,
        entry_threshold=2.0, # Z-score threshold to enter a trade (e.g., go long the undervalued asset and short the overvalued)
        exit_threshold=0.0,  # Z-score threshold to exit a trade (e.g., when the spread reverts to its mean)
        lookback=30,         # Number of past days to consider for calculating the rolling mean and standard deviation of the spread
        transaction_cost=0.001 # Cost per trade as a percentage of the trade value
    )

    # Plot the overall portfolio performance curve
    pairs_trading.plot_portfolio_performance()

    # Print the overall portfolio performance metrics
    pairs_trading.portfolio_performance_metrics()

    # Iterate through each backtested pair and plot individual performance metrics and charts
    for pair, result in pairs_trading.results.items():
        print(f"\nAnalyzing pair: {pair[0]}-{pair[1]}")
        pairs_trading.plot_closing_prices(result) # Plot closing prices of the pair
        print("\n")
        pairs_trading.plot_rel_close_price(result) # Plot normalized relative closing prices
        print("\n")
        pairs_trading.plot_zscore(result) # Plot the z-score of the spread and trading signals
        print("\n")
        pairs_trading.plot_long_short_positions(result) # Plot the long/short positions taken
        print("\n")
        pairs_trading.plot_PnL(result) # Plot PnL and cumulative returns
        print("\n")
        pairs_trading.plot_portfolio_curve(result) # Plot the portfolio value for this specific pair
        print("\n")
        pairs_trading.plot_drawdown(result) # Plot the drawdown for this specific pair
        # print("\n") # Commented out an extra newline from the original notebook
        pairs_trading.performance_metrics(result) # Print individual pair performance metrics
        print("\n\n") # Add extra newlines for separation in the output

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
