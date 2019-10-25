"""
    https://www.kaggle.com/chirag19/time-series-analysis-with-python-beginner
    https://www.kaggle.com/hsankesara/time-series-analysis-and-forecasting-using-arima
    https://www.kaggle.com/txtrouble/time-series-analysis-with-python

    To READ only:
    https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python
"""

import numpy as np 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParamas['figure.figsize'] = 15,6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

#Loading the data 
data = pd.read_csv(r'D:\C++\PYTHON\ml\AirPassengers.csv')
""" Data has the passenger number of each month from 1949-1960"""
"""
    >> data.head()
       Month  #Passengers
0 1949-01-01          112
1 1949-02-01          118
2 1949-03-01          132
3 1949-04-01          129
4 1949-05-01          121

"""

#converts the data into date time format
data['Month'] = pd.to_datetime(data.Month)
""" 
       Month  #Passengers
0 1949-01-01          112
1 1949-02-01          118
2 1949-03-01          132
3 1949-04-01          129
4 1949-05-01          121"""
data = data.set_index(data.Month)
# index the data with the months 
"""
                   Month  #Passengers
Month
1949-01-01 1949-01-01          112
1949-02-01 1949-02-01          118
1949-03-01 1949-03-01          132
1949-04-01 1949-04-01          129
1949-05-01 1949-05-01          121 """

data.drop('Month', axis = 1, inplace = True)
# drops the month column of the data
data.head()

""" 
                #Passengers
Month
1949-01-01          112
1949-02-01          118
1949-03-01          132
1949-04-01          129
1949-05-01          121 """

#Indexing the data is helpful and makes it convinient to handle data 

ts = data['#Passengers']
plt.plot(ts)
plt.show()

plt.subplot(221)
plt.hist(ts)
plt.subplot(222)
# kde is the kernal density plot used to represnt the probability density function of a random variable
ts.plot(kind='kde')
plt.show()

""" Now the rolling mean and the rolling standard deviation is calcuated for the time series"""

def test_stationarity(timeseries):
    # calculating the rolling mean and standard deviations
    rolmean = timeseries.rolling(window=12).mean()
    rolsd = timeseries.rolling(window=12).std()
    # plotting the rolling statistics
    orig = plt.plot(timeseries, color ='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='ROlling Mean')
    sd = plt.plot(rolsd, color='black', label='Rolling SD')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()

print('Results of Dickey_fuller Test:')
dftest = adfuller(ts)
dfoutput = pd.Series(dftest[0:4], index = ['Test Statitics', 'p-value','Numberoof Observations Used'])
for key, values in dftest[4].items():
    dfoutput['Critical Value (%s)' %key] = value
print(dfoutput)

"""Its almost impossible to make a series perfectly stationary, but we try to take it as close as possible.

There are 2 major reasons behind non-stationaruty of a TS:

Trend – varying mean over time. For eg, in this case we saw that on average, the number of passengers was growing over time.
Seasonality – variations at specific time-frames. eg people might have a tendency to buy cars in a particular month because of pay increment or festivals.

The underlying principle is to model or estimate the trend and seasonality in the series and remove those from the series to get a stationary 
series. Then statistical forecasting techniques can be implemented on this series. The final step would be to convert the forecasted values 
into the original scale by applying trend and seasonality constraints back."""

# Estimating and eliminating trend

"""we can apply transformation which penalize higher values more than smaller values. 
These can be taking a log, square root, cube root, etc. Lets take a log transform here for simplicity."""

ts_log = np.log(ts)
plt.plot(ts_log)
plt.show()

""" we can use some techniques to estimate or model this trend and then remove it from the series. There can be many ways of doing it and some of most commonly used are:

Aggregation – taking average for a time period like monthly/weekly averages
Smoothing – taking rolling averages
Polynomial Fitting – fit a regression model

We will apply smoothing here."""

moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

# we can see an increasing trend

ts_log_moving_avg_diff = ts_log- moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

"""  After taking the log and then calulating the rolling average
    and subtracting the rolling average from the time series
    we can see that the increasing trend is removed from the time series
    The trend can never be removed 100% but it is visible that there is no 
    specific trend.
    """
""" There are other methods for caluating the rolling average for better accuracy.
    These include weighted moving averge (more recent values are given higher weights),
     exponential weighted moving average( weights are assigned to all the previous values
     with a decay factor)"""


exp_weighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(exp_weighted_avg, color='pink')
plt.plot(moving_avg, color='purple')
plt.show()

ts_log_ema_diff = ts_log- exp_weighted_avg
test_stationarity(ts_log_ema_diff)

""" Eliminating trend and Seasonality:
    Two methods:

Differencing (taking the differece with a particular time lag)
Decomposition (modeling both trend and seasonality 
and removing them from the model)"""

#Differencing:
ts_log_diff_shift = ts_log - ts_log.shift()
plt.plot(ts_log_diff_shift)
plt.show()
ts_log_diff_shift.dropna(inplace=True)
test_stationarity(ts_log_diff_shift)

#DEcomposing

decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label ='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label ='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label ='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label ='Residual')
plt.legend(loc='best')
plt.show()


# the residual is the timeseries after the removal of trend and seasonality

ts_log_decompose = residual 
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
