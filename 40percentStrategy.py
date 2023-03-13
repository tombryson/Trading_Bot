import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasell_frequency
data = pd.read_csv('data.csv', parse_dates=['date'])
data['week'] = data['date']
data = data.groupby('week')['Adj Close'].last().reset_index()

# Plot time series data
plt.plot(data['week'], data['Adj Close'])
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.title('Time Series Data')
plt.savefig('figure1.png')

def strategy_1(data, buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage):
    # Initialize variables
    bank_reserve=1000
    buy_week_count = 0 # Counter for number of consecutive weeks below regression line
    sell_week_count = 0 # Counter for number of consecutive weeks above predicted price
    buy_dates = []
    sell_dates = []
    buy_list = []
    total_profit = 0
    purchase_cost = 0 # Define variable outside the if-else block
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
        # Add $3000 to bank_reserve every 6 weeks
        if i % 6 == 0:
            bank_reserve += 300

        # Check if price is below regression line
        price_above_regression = current_price >= regression[i]
        if not price_above_regression:
            buy_week_count += 1
            sell_week_count = 0

            if buy_week_count > buy_frequency_limit and np.any(current_price < regression[i] - buy_threshold * std):
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
                    tax_rate = 0.25 if holding_period > 52 else 0.5
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

    # Print results
    return (total_profit, bank_reserve, buy_dates, sell_dates, sell_threshold, buy_threshold, regression, std, buy_list, coeffs, total_cgt)

# total_profit, bank_reserve, buy_dates, sell_dates, sell_threshold, buy_threshold, regression, std, buy_list, coeffs, total_cgt = strategy_1(data, buy_frequency_limit=5, buy_threshold=1, sell_threshold=1, sell_frequency_limit=8, sell_percentage=0.4)

#### Testing ###################################################################
import random
import math
from datetime import datetime

# Define the range of values for each input parameter
buy_frequency_limit_range = [i / 100 for i in range(300, 1100)]
buy_threshold_range = [i / 100 for i in range(20, 180)]

sell_frequency_limit_range = [i / 100 for i in range(500, 1200)]
sell_threshold_range = [i / 100 for i in range(20, 180)]

sell_percentage_range = [i / 100 for i in range(10, 90)]

# Define the number of iterations to run the search algorithm
num_iterations = 10000

# Define a function to generate a random combination of input parameters
def generate_random_params():
    buy_frequency_limit = random.choice(buy_frequency_limit_range)
    buy_threshold = random.choice(buy_threshold_range)
    sell_threshold = random.choice(sell_threshold_range)
    sell_frequency_limit = random.choice(sell_frequency_limit_range)
    sell_percentage = random.choice(sell_percentage_range)
    return buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage

def run_strategy(params):
    buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage = params
    # Call the strategy function with the given input parameters
    total_profit = strategy_1(data, buy_frequency_limit=buy_frequency_limit, buy_threshold=buy_threshold, sell_threshold=sell_threshold, sell_frequency_limit=sell_frequency_limit, sell_percentage=sell_percentage)[0]
    print(f"Total Profit: ${round(total_profit):,}")
    return total_profit

def acceptance_probability(old_profit, new_profit, temperature):
    if new_profit > old_profit:
        return 1.0
    else:
        return math.exp((new_profit - old_profit) / temperature)

# Define a function to run the simulated annealing algorithm
def run_simulated_annealing(num_iterations):
    # Set initial temperature and cooling rate
    temperature = 1.0
    cooling_rate = 0.002
    # Generate an initial set of parameters and calculate the initial profit
    current_params = generate_random_params()
    current_profit = run_strategy(current_params)
    # Set the best parameters and profit to the current ones
    best_params = current_params
    best_profit = current_profit
    # Loop over the specified number of iterations
    for i in range(num_iterations):
        # Generate a new set of parameters
        new_params = generate_random_params()
    
        # Calculate the profit using the new parameters
        new_profit = run_strategy(new_params)
        
        # Calculate the acceptance probability for the new parameters
        accept_prob = acceptance_probability(current_profit, new_profit, temperature)
        
        # If the new parameters are better, accept them
        if accept_prob > random.random():
            current_params = new_params
            current_profit = new_profit
        
        # If the new profit is better than the best profit, update the best parameters and profit
        if new_profit > best_profit:
            best_params = new_params
            best_profit = new_profit
        
        # Reduce the temperature according to the cooling rate
        temperature *= 1 - cooling_rate
    
    # Return the best parameters and profit
    return best_params, best_profit

# Run the randomized search algorithm and print the best result
best_params, best_profit = run_simulated_annealing(num_iterations)

print(f"Best profit:  ${round(best_profit):,} with parameters: \n\nbuy_frequency_limit={best_params[0]}, buy_threshold={best_params[1]}, sell_threshold={best_params[2]}, sell_frequency_limit={best_params[3]}, sell_percentage={best_params[4]} \n")

# Pass the best result to Data Analysis

total_profit, bank_reserve, buy_dates, sell_dates, sell_threshold, buy_threshold, regression, std, buy_list, coeffs, total_cgt = strategy_1(data, buy_frequency_limit=best_params[0], buy_threshold=best_params[1], sell_threshold=best_params[2], sell_frequency_limit=best_params[3], sell_percentage=best_params[4])


# DATA ANALYSIS ###############################################################

# Plot the buy and sell markers on the time series plot
# Add regression line to plot
plt.plot(data['week'], regression, '-', color='blue', label='Regression Line')
plt.plot(buy_dates, data.loc[data['week'].isin(buy_dates), 'Adj Close'], 'go', label='BUY')
plt.plot(sell_dates, data.loc[data['week'].isin(sell_dates), 'Adj Close'], 'ro', label='SELL')
plt.plot(data['week'], regression + sell_threshold * std, '--', color='orange', label='SELL THRESHOLD')
plt.plot(data['week'], regression - buy_threshold * std, '--', color='teal', label='BUY THRESHOLD')

# Add legend to the plot
plt.legend()
# Show the plot
plt.savefig('figure2.png')


# Calculate Variables
year = datetime.strptime(str(buy_dates[0]), '%Y-%m-%d %H:%M:%S').year
value_in_market = 0
contributions = 1000 + (10 * 1200)
while len(buy_list) != 0:
    buy_price, purchase_week, shares = buy_list.pop()
    value_in_market += shares * buy_price
ending_value = total_profit + contributions + value_in_market
number_of_years = 2023 - year
annualized_return = ((ending_value / contributions) ** (1/number_of_years)) - 1
number_of_years = 2023 - year

# Print Calculations
print(f"{number_of_years} years in the market with First buy in {year}")
print(f'Total of {number_of_years} years of Contributions: ${contributions:,}')
print('Total Profit: ${:,}'.format(math.ceil(total_profit)))
print(f"Value still in market: ${round(value_in_market):,}")
print(f"Amount in bank: ${round(bank_reserve):,}")
print(f"Total Revenue is: ${round(ending_value):,}")
print(f"Total Tax: ${round(total_cgt):,}")
print(f"{round(annualized_return*100,2)}% Return y/y")
print(f"Market Return: {round(coeffs[0] * 100)}%")