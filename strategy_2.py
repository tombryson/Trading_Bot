import pandas as pd
import numpy as np
import math as math

# Load datasell_frequency
data = pd.read_csv('data.csv', parse_dates=['date'])
data['week'] = data['date']
data = data.groupby('week')['Adj Close'].last().reset_index()

def strategy_2(data, buy_frequency_limit):
    # Initialize variables
    bank_reserve = 1000
    initial_value = bank_reserve # Used for yy calculations
    buy_week_count = 0 # Counter for number of consecutive weeks below regression line
    buy_dates = []
    buy_list = []
    contributions = 0
    twr_log_numerator = 0
    twr_log_denominator = 0
    total_profit = 0
    total_cgt = 0

    # Calculate regression and standard deviation
    x = np.arange(len(data))
    y = data['Adj Close'].values
    coeffs = np.polyfit(x, y, deg=1)
    regression = np.poly1d(np.polyfit(x, y, 1))(x)
    std = np.std(y - regression)

    # Loop over each week
    for i in range(1, len(data)):
        current_price = data['Adj Close'][i]
        buy_week_count += 1
        
        # Add $3000 to bank_reserve every 3 months
        if i % 13 == 0:
            portfolio_value = bank_reserve
            dividend_payment = 0
            for buy_price, purchase_week, shares in buy_list:
                value = shares * current_price
                portfolio_value += value
                dividend_payment += value * 0.01 # Incorporate a 4% p.a dividend
            bank_reserve += dividend_payment
            bank_reserve += 3000
            contributions += 3000
            portfolio_value += 3000 + dividend_payment

        if buy_week_count > buy_frequency_limit:
            # Calculate portfolio value before buying
            portfolio_value = bank_reserve
            for buy_price, purchase_week, shares in buy_list:
                portfolio_value += shares * current_price

            twr_log_numerator += math.log(portfolio_value)
            # Buy as many shares as possible with the bank_reserve amount
            share_price = current_price
            shares_to_buy = bank_reserve / share_price
            purchase_cost = shares_to_buy * share_price + 10 ## Fixed cost for brokerage
            # Update buy_list
            buy_list.append((share_price, i, shares_to_buy))
            # Update variables
            bank_reserve -= purchase_cost
            # Add the buy date to the list of buy dates
            buy_dates.append(data['week'][i])
            buy_week_count = 0

            # Calculate portfolio value after buying
            portfolio_value = bank_reserve
            for buy_price, purchase_week, shares in buy_list:
                portfolio_value += shares * current_price

            twr_log_denominator += math.log(portfolio_value)

    revenue = portfolio_value

    # After the loop, calculate the TWR and y/y return
    twr = math.exp((twr_log_numerator - twr_log_denominator) / (len(data) / 52)) - 1
    yy_return = (revenue - initial_value) / initial_value

    # Print results
    return (bank_reserve, buy_dates, regression, std, buy_list, coeffs, contributions, revenue, twr, yy_return, total_profit, total_cgt)