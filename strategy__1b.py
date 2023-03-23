import pandas as pd
import numpy as np

# Load datasell_frequency
data = pd.read_csv('data.csv', parse_dates=['date'])
data['week'] = data['date']
data = data.groupby('week')['Adj Close'].last().reset_index()

def strategy_1(data, buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage):
    # Initialize variables
    bank_reserve = 1000
    buy_week_count = 0 # Counter for number of consecutive weeks below regression line
    sell_week_count = 0 # Counter for number of consecutive weeks above predicted price
    buy_dates = []
    sell_dates = []
    buy_list = []
    total_profit = 0
    purchase_cost = 0 # Define variable outside the if-else block
    total_cgt = 0
    contributions = 0

    # Calculate regression and standard deviation
    x = np.arange(len(data))
    y = data['Adj Close'].values
    coeffs = np.polyfit(x, y, deg=1)
    regression = np.poly1d(np.polyfit(x, y, 1))(x)
    std = np.std(y - regression)

    # Loop over each week
    for i in range(1, len(data)):
        current_price = data['Adj Close'][i]
        # Add $3000 to bank_reserve every 6 weeks
        if i % 13 == 0:
            bank_reserve += 3000
            contributions += 3000

        # Check if price is below regression line
        price_above_regression = current_price >= regression[i]
        if not price_above_regression:
            buy_week_count += 1
            sell_week_count = 0

            if buy_week_count > buy_frequency_limit and np.any(current_price < regression[i] - buy_threshold * std):
                # Buy as many shares as possible with the bank_reserve amount
                share_price = current_price
                shares_to_buy = bank_reserve // share_price
                purchase_cost = shares_to_buy * share_price + 10 ## Fixed cost for brokerage

                # Update buy_list
                buy_list.append((share_price, i, shares_to_buy))

                # Update variables
                bank_reserve -= purchase_cost

                # Add the buy date to the list of buy dates
                buy_dates.append(data['week'][i])
                buy_week_count = 0

        # Price is above regression line
        else:
            buy_week_count = 0
            sell_week_count += 1

            if sell_week_count > sell_frequency_limit and np.any(current_price >= regression[i] + sell_threshold * std) and len(buy_list) >= 1:

                # Calculate total shares in the market
                shares_to_sell = 0
                shares_in_market = buy_list.copy()
                while len(shares_in_market) > 0:
                    buy_price, purchase_week, shares = shares_in_market.pop()
                    shares_to_sell += shares
                
                # Sell percent adjusted portion of invested shares
                shares_to_sell = int(shares_to_sell * sell_percentage)

                # Set up variables for selling 
                sell_price = current_price
                sale_revenue = shares_to_sell * sell_price

                #Determine CGT
                while shares_to_sell > 0 and buy_list:
                    buy_price, purchase_week, shares = buy_list.pop()
                    holding_period = i - purchase_week
                    cg_per_share = sell_price - buy_price
                    partial_profit = cg_per_share * shares - 10 # Fixed fee for brokerage
                    tax_rate = 0.16 if holding_period > 52 else 0.32
                    if shares_to_sell >= shares:
                        partial_profit = cg_per_share * shares
                        cgt = partial_profit * tax_rate
                        shares_to_sell -= shares
                    else:
                        partial_profit = cg_per_share * shares_to_sell
                        cgt = partial_profit * tax_rate
                        unsold_shares = shares - shares_to_sell
                        buy_list.append((buy_price, purchase_week, unsold_shares))
                        shares_to_sell = 0
                    total_profit += partial_profit
                    partial_profit = 0
                    total_cgt += cgt

                    # Remove tax from reserve and Reset CGT
                    bank_reserve -= cgt
                    cgt = 0

                # Update variables
                bank_reserve += sale_revenue
                sell_week_count = 0

                # Add the sell date to the list of sell dates
                sell_dates.append(data['week'][i])

    # Determine the value still in the market
    shares_in_market = buy_list.copy()
    final_price = data['Adj Close'][-1]
    while len(shares_in_market) > 0:
        buy_price, purchase_week, shares = shares_in_market.pop()
        value = shares * final_price

    revenue = bank_reserve + value
    # Print results
    return (total_profit, bank_reserve, buy_dates, sell_dates, sell_threshold, buy_threshold, regression, std, buy_list, coeffs, total_cgt, contributions, revenue)
