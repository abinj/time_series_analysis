import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


def mean_average_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'SeaPlaneTravel.csv'
data = pd.read_csv(DATA_PATH)
print(data.head(10))

# Exploratory Data Analysis
plt.figure(figsize=(17, 8))
plt.plot(data.Passengers)
plt.title('Passengers Traffic')
plt.ylabel('Number of passengers')
plt.xlabel('day')
plt.grid(False)
plt.show()


def plot_moving_average(series, window, plot_intervals=False, scale=1.96):
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')

    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

#Smooth by the previous 5 days (by week)
plot_moving_average(data.Passengers, 5)

#Smooth by the previous month (30 days)
plot_moving_average(data.Passengers, 30)

#Smooth by previous quarter (90 days)
plot_moving_average(data.Passengers, 90, plot_intervals=True)


