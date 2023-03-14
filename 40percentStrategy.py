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

# total_profit, bank_reserve, buy_dates, sell_dates, sell_threshold, buy_threshold, regression, std, buy_list, coeffs, total_cgt = strategy_1(data, buy_frequency_limit=5, buy_threshold=1, sell_threshold=1, sell_frequency_limit=8, sell_percentage=0.4)

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

def run_strategy(params):
    buy_frequency_limit, buy_threshold, sell_threshold, sell_frequency_limit, sell_percentage = params
    # Call the strategy function with the given input parameters
    revenue = strategy_1(data, buy_frequency_limit=buy_frequency_limit, buy_threshold=buy_threshold, sell_threshold=sell_threshold, sell_frequency_limit=sell_frequency_limit, sell_percentage=sell_percentage)[0]
    # total_profit = strategy_2(data)
    print(f"Total Revenue: ${round(revenue):,}")
    return revenue

def acceptance_probability(old_revenue, new_revenue, temperature):
    if new_revenue > old_revenue:
        return 1.0
    else:
        return math.exp((new_revenue - old_revenue) / temperature)

# Define a function to run the simulated annealing algorithm
def run_simulated_annealing(num_iterations):
    # Set initial temperature and cooling rate
    temperature = 1.0
    cooling_rate = 0.003
    # Generate an initial set of parameters and calculate the initial revenue
    current_params = generate_random_params()
    current_revenue = run_strategy(current_params)
    # Set the best parameters and profit to the current ones
    best_params = current_params
    best_revenue = current_revenue
    # Loop over the specified number of iterations
    for i in range(num_iterations):
        # Generate a new set of parameters
        new_params = generate_random_params()
    
        # Calculate the profit using the new parameters
        new_revenue = run_strategy(new_params)
        
        # Calculate the acceptance probability for the new parameters
        accept_prob = acceptance_probability(current_revenue, new_revenue, temperature)
        
        # If the new parameters are better, accept them
        if accept_prob > random.random():
            current_params = new_params
            current_revenue = new_revenue
        
        # If the new profit is better than the best profit, update the best parameters and profit
        if new_revenue > best_revenue:
            best_params = new_params
            best_revenue = new_revenue
        
        # Reduce the temperature according to the cooling rate
        temperature *= 1 - cooling_rate
    
    # Return the best parameters and profit
    return best_params, best_revenue

# Run the randomized search algorithm and print the best result
best_params, best_revenue = run_simulated_annealing(num_iterations)

print(f"Best revenue:  ${round(best_revenue):,} with parameters: \n\nbuy_frequency_limit={best_params[0]}, buy_threshold={best_params[1]}, sell_threshold={best_params[2]}, sell_frequency_limit={best_params[3]}, sell_percentage={best_params[4]} \n")

# Pass the best result to Data Analysis

total_profit, bank_reserve, buy_dates, sell_dates, sell_threshold, buy_threshold, regression, std, buy_list, coeffs, total_cgt, contributions, revenue = strategy_1(data, buy_frequency_limit=best_params[0], buy_threshold=best_params[1], sell_threshold=best_params[2], sell_frequency_limit=best_params[3], sell_percentage=best_params[4])

# total_profit, bank_reserve, buy_dates, buy_list = strategy_2(data)

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
while len(buy_list) != 0:
    buy_price, purchase_week, shares = buy_list.pop()
    value_in_market += shares * buy_price
ending_value = total_profit + contributions + value_in_market
number_of_years = 10
annualized_return = ((ending_value / contributions) ** (1/number_of_years))
#### year_on_year_return =  ((ending_value - beginning_value) / beginning_value) * 100



ending_value = round(ending_value)
print(ending_value)
print(f"{contributions}")
print(f"{number_of_years}")
TWR = ending_value / (contributions) ** (1 / 10) - 1
print(f"TWR is {TWR}")


print(f"{annualized_return}")

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