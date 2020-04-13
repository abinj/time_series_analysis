from statsmodels.tsa.stattools import adfuller
from numpy import log
import pandas as pd

#
DATA_PATH = 'SeaPlaneTravel.csv'
data = pd.read_csv(DATA_PATH)
# result = adfuller(data)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])

print(data.Passengers)

# data = np.array([[1, 2, 2], [3, 3, 3],[4, 4, 4]])