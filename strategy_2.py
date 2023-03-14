import pandas as pd
import numpy as np

# Load datasell_frequency
data = pd.read_csv('data.csv', parse_dates=['date'])
data['week'] = data['date']
data = data.groupby('week')['Adj Close'].last().reset_index()

def strategy_2(data, buy_frequency_limit):
    # Initialize variables
    bank_reserve = 1000
    buy_week_count = 0 # Counter for number of consecutive weeks below regression line
    buy_dates = []
    buy_list = []
    total_profit = 0
    contributions = 0

    # Calculate regression and standard deviation
    x = np.arange(len(data))
    y = data['Adj Close'].values
    coeffs = np.polyfit(x, y, deg=1)
    regression = np.poly1d(np.polyfit(x, y, 1))(x)
    std = np.std(y - regression)

    # Loop over each week
    for i in range(1, len(data)):
        # Add $3000 to bank_reserve every 6 weeks
        if i % 13 == 0:
            bank_reserve += 3000
            contributions += 3000
            buy_week_count += 1

            if buy_week_count > buy_frequency_limit:
                # Buy as many shares as possible with the bank_reserve amount
                share_price = data['Adj Close'][i]
                shares_to_buy = bank_reserve // share_price
                purchase_cost = shares_to_buy * share_price + 10 ## Fixed cost for brokerage
                # Update buy_list
                buy_list.append((share_price, i, shares_to_buy))
                # Update variables
                bank_reserve -= purchase_cost
                # Add the buy date to the list of buy dates
                buy_dates.append(data['week'][i])
                buy_week_count = 0

    # Print results
    return (total_profit, bank_reserve, buy_dates, regression, std, buy_list, coeffs, contributions)