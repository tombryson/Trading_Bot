import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy_1 import strategy_1
from strategy_2 import strategy_2

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


#### Testing ###################################################################
import random
import math
from datetime import datetime

# Define the range of values for each input parameter
buy_frequency_limit_range = [i / 100 for i in range(200, 1300)]
buy_threshold_range = [i / 100 for i in range(20, 280)]
sell_frequency_limit_range = [i / 100 for i in range(200, 1400)]
sell_threshold_range = [i / 100 for i in range(20, 280)]
sell_percentage_range = [i / 100 for i in range(10, 90)]

# Define the number of iterations to run the search algorithm
num_iterations = 1000

# Define a function to generate a random combination of input parameters
def generate_random_params():
    buy_frequency_limit = np.random.choice(buy_frequency_limit_range)
    buy_threshold = np.random.choice(buy_threshold_range)
    sell_threshold = np.random.choice(sell_threshold_range)
    sell_frequency_limit = np.random.choice(sell_frequency_limit_range)
    sell_percentage = np.random.choice(sell_percentage_range)
    return buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage

<<<<<<< HEAD:40percentStrategy.py
def run_strategy(params):
    buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage = params

    # Call the strategy function with the given input parameters

    revenue = strategy_1(data, buy_frequency_limit=buy_frequency_limit, buy_threshold=buy_threshold, sell_threshold=sell_threshold, sell_frequency_limit=sell_frequency_limit, sell_percentage=sell_percentage)[0]
    
    # revenue = strategy_2(data, buy_frequency_limit=buy_frequency_limit)[0]

    print(f"Total Revenue: ${round(revenue):,}")
=======
def run_strategy(params, strategy_func):
    if strategy_func == strategy_1:
        buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage = params
        result = strategy_1(data, buy_frequency_limit=buy_frequency_limit, buy_threshold=buy_threshold, sell_threshold=sell_threshold, sell_frequency_limit=sell_frequency_limit, sell_percentage=sell_percentage)
        revenue = result[0] + result[1] + result[11]
        print(f"Total Revenue (Strategy 1): ${round(revenue):,}")
    elif strategy_func == strategy_2:
        buy_frequency_limit = params
        result = strategy_2(data, buy_frequency_limit=buy_frequency_limit)
        revenue = result[0] + result[1] + result[7]
        print(f"Total Revenue (Strategy 2): ${round(revenue):,}")
    else:
        raise ValueError("Invalid strategy function provided.")
>>>>>>> 8a8a5a4 (useless files):Stock_bot.py
    return revenue


def acceptance_probability(old_revenue, new_revenue, temperature):
    if new_revenue > old_revenue:
        return 1.0
    else:
        return math.exp((new_revenue - old_revenue) / temperature)

# Define a function to run the simulated annealing algorithm
def run_simulated_annealing(num_iterations, strategy):
    temperature = 1.0
    cooling_rate = 0.003
    current_params = generate_random_params()
    current_revenue = run_strategy(current_params, strategy)
    best_params = current_params
    best_revenue = current_revenue
    for i in range(num_iterations):
        new_params = generate_random_params()
        new_revenue = run_strategy(new_params, strategy)
        accept_prob = acceptance_probability(current_revenue, new_revenue, temperature)
        if accept_prob > random.random():
            current_params = new_params
            current_revenue = new_revenue
        if new_revenue > best_revenue:
            best_params = new_params
            best_revenue = new_revenue
        temperature *= 1 - cooling_rate
    return best_params, best_revenue

# Run the randomized search algorithm and print the best result
best_params, best_revenue = run_simulated_annealing(num_iterations, strategy_1)

print(f"Best revenue:  ${round(best_revenue):,} with parameters: \n\nbuy_frequency_limit={best_params[0]}, buy_threshold={best_params[1]}, sell_threshold={best_params[2]}, sell_frequency_limit={best_params[3]}, sell_percentage={best_params[4]} \n")


total_profit, bank_reserve, buy_dates, sell_dates, sell_threshold, buy_threshold, regression, std, buy_list, coeffs, total_cgt, contributions, revenue, twr, yy_return = strategy_1(data, buy_frequency_limit=best_params[0], buy_threshold=best_params[1], sell_threshold=best_params[2], sell_frequency_limit=best_params[3], sell_percentage=best_params[4])

# Strategy 2
# bank_reserve, buy_dates, regression, std, buy_list, coeffs, contributions, revenue = strategy_2(data, buy_frequency_limit=best_params[0])


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
ending_value = revenue
number_of_years = 10
annualized_return = ((ending_value / contributions) ** (1/number_of_years))
#### year_on_year_return =  ((ending_value - beginning_value) / beginning_value) * 100

ending_value = round(ending_value)
print(ending_value)
print(f"{contributions}")
print(f"{number_of_years}")
print(f"TWR is {twr}")
print(f"{annualized_return}")

# Print Calculations
print(f"{number_of_years} years in the market with First buy in {year}")
print(f'Total of {number_of_years} years of Contributions: ${contributions:,}')
print('Total Profit: ${:,}'.format(math.ceil(total_profit)))
print(f"Value still in market: ${round(value_in_market):,}")
print(f"Amount in bank: ${round(bank_reserve):,}")
print(f"Total Revenue is: ${round(ending_value):,}")
print(f"Total Tax: ${round(total_cgt):,}")
print(f"{yy_return}% Return y/y")
print(f"Market Return: {round(coeffs[0] * 100)}%")