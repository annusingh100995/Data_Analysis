import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import plotly.graph_objects as go


def Linear_Regression_for_campaign(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    model = LinearRegression().fit(X_train, y_train) 
    print("Regressor intercept : " , regressor.intercept_)
    #For retrieving the slope:
    print("Regressor Coefficient : ", regressor.coef_)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    #df
    print('Coefficient of determination:',model.score(X_train, y_train))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    #df1 = df.head(25)
    #df1.plot(kind='bar',figsize=(16,10))
    #plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    #plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    #plt.show()
    plt.scatter(X_test, y_test,  color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()

data2 = pd.read_csv(r'D:\C++\PYTHON\ml\trivago_market_intelligence\1921 - Data Analyst - MINT\session_data.csv', sep=';')
data2['sess_start_time'] = pd.to_datetime(data2['session_start_text'] , format= '%H:%M:%S' ).dt.time
data2['sess_end_time'] = pd.to_datetime(data2['session_end_text'] , format= '%H:%M:%S' ).dt.time


t2 = pd.to_datetime(data2['session_end_text'])
t1 = pd.to_datetime(data2['session_start_text'])
d =t2 - t1
a = d.astype('timedelta64[s]')
data2['Time_spent_on_session_in_seconds'] = a

data = data2[['session','sess_start_time','sess_end_time','clickouts','Time_spent_on_session_in_seconds','booking']]
data_sorted = data.sort_values(by = 'sess_start_time', ascending=True)


# Booking VS Start Time
plt.step(data_sorted.sess_start_time ,data_sorted['booking'] )
plt.xlabel('Session Start Time')
plt.ylabel('Booking')
plt.title('Booking vs Start Time')
plt.savefig('Booking vs Start Time')
plt.show()

# Clickouts vs Start Time
plt.step(data_sorted.sess_start_time ,data_sorted['clickouts'] )
plt.xlabel('Session Start Time')
plt.ylabel('Clickouts')
plt.title('Clickouts vs Start Time')
plt.savefig('Clickouts vs Start Time')
plt.show()



# Booking vs Start time zoomed in 
fig, axs = plt.subplots(5)
fig.suptitle('Booking vs Start Time')
for y , i in zip( range(0,5)  , (range(0,8001,2000)) ):
    axs[y].step(data_sorted.iloc[i:i+2000,:].sess_start_time ,data_sorted.iloc[i:i+2000,:]['booking'] )


plt.savefig('Booking vs Time')
plt.show()
# I don't see any specific pattern here. It is not evident that there is some time where the booking are more comman.

# Clickouts vs Start time zoomed in 
fig, axs = plt.subplots(5)
fig.suptitle('Clickouts vs Start Time')
for y , i in zip( range(0,5)  , (range(0,8001,2000)) ):
    axs[y].step(data_sorted.iloc[i:i+2000,:].sess_start_time ,data_sorted.iloc[i:i+2000,:]['clickouts'] )


plt.savefig('Clickouts vs Time Zoomed')
plt.show()

# I see that there are 2-5 clcikouts on avarage

# Booking vs Time spent in 
fig, axs = plt.subplots(5)
fig.suptitle('Booking vs Time Spent in Seconds')
for y , i in zip( range(0,5)  , (range(0,8001,2000)) ):
    axs[y].step(abs(data_sorted.iloc[i:i+2000,:].Time_spent_on_session_in_seconds) ,data_sorted.iloc[i:i+2000,:]['booking'] )


plt.savefig('Booking vs Time Spent in seconds')
plt.show()

# Finding corelation between Booking and Time spent on the session and 
# clickouts and time spent on sessions

data_sorted['Time_spent_on_session_in_seconds'].corr(data_sorted['booking'])
# 0.012008442677019558

data_sorted['Time_spent_on_session_in_seconds'].corr(data_sorted['clickouts'])
# 0.009441968499793372

# There in not much relation between time spent and booking or clickouts


# Clicksouts vs booking
sns.catplot(x="clickouts", y="booking",  kind="swarm" ,data=data_sorted)
plt.show()

# Separating data into two groups based on booking
data_0 = data_sorted[data_sorted['booking']==0]
data_1 = data_sorted[data_sorted['booking']==1]

sns.catplot(x="clickouts", y="booking", data=data_1)
sns.catplot(x="clickouts", y="booking",  kind="swarm" ,data=data_1)

# Checking number of bookings for different clickout values
booking_for_clickouts_0 = data_1.groupby(['clickouts']).size()
print(print("Clickouts for negative booking : ", booking_for_clickouts_1))

booking_for_clickouts_1 = data_0.groupby(['clickouts']).size()
print("Clickouts for positive booking : ", booking_for_clickouts_1)


# Fitting LR for data where booking = 1

y_data_sorted = data_sorted.booking.values.reshape(-1,1)
X_data_sorted = data_sorted.clickouts.values.reshape(-1,1)
Linear_Regression_for_campaign(X_data_sorted, y_data_sorted)

"""
>>> Linear_Regression_for_campaign(X_data_sorted, y_data_sorted)
Regressor intercept :  [0.12800077]
Regressor Coefficient :  [[-0.01264203]]
Coefficient of determination: 0.0020776542274185683
Mean Absolute Error: 0.1749910839374949
Mean Squared Error: 0.08764412122914908
Root Mean Squared Error: 0.29604749826531057

"""
# Time spent vs booking
y_data_sorted_booking = data_sorted.booking.values.reshape(-1,1)
X_data_sorted_time_spent = data_sorted.Time_spent_on_session_in_seconds.values.reshape(-1,1)
Linear_Regression_for_campaign(X_data_sorted_time_spent, y_data_sorted_booking)
"""
Regressor intercept :  [0.09642891]
Regressor Coefficient :  [[1.13980669e-06]]
Coefficient of determination: 0.00015279802121559438
Mean Absolute Error: 0.17519754891293188
Mean Squared Error: 0.08798504369310657
Root Mean Squared Error: 0.2966227295624976
"""

# There is a positive corelation between the time spent on the site and booking. But the 
# coefficient of determination is very low.

y_data_sorted_booking = data_sorted.booking.values.reshape(-1,1)
X_data_sorted_sess_start_time = data_sorted.sess_start_time.values.reshape(-1,1)
Linear_Regression_for_campaign(X_data_sorted_sess_start_time, y_data_sorted_booking)


for i in range(0,10000,500):
    count = len(data_sorted.iloc[i:i+500,:][data_sorted.iloc[i:i+500,:]['booking']==1])
    print("Bin Count ", count)
    #data_sorted.iloc[i, [1,2]])


# Dividing the data into 24 groups. One group for each hour of the day.
df = data_sorted.groupby(data_sorted['sess_start_time'].map(lambda x: x.hour))

count = data_sorted.groupby(data_sorted['sess_start_time'].map(lambda x: x.hour))['booking'].value_counts()

a = pd.DataFrame(count)

# List for counts where booking == 1 for each hour of the day.
count_per_hour_1 =[]
for i in range(0,24):
    count_per_hour_1.append(a['booking'][i][1])

hour_in_day = pd.Series(range(0,24))
booking_per_hour = pd.Series(count_per_hour_1)

# Pie chart
(booking_per_hour).plot.pie(labels= range(0,24), autopct='%.2f', fontsize=10, figsize=(6, 6))
plt.title('Booking per Hour')
plt.savefig('Booking per Hour pei chart')
plt.show()

plt.plot(hour_in_day , booking_per_hour)
plt.title('Booking per Hour')
plt.xlabel('Time (0 HRS - 23 HRS)')
plt.ylabel('Number of bookings')
plt.xticks(hour_in_day)
plt.yticks(booking_per_hour)
plt.savefig('Number of booking vs Hour of Day')
plt.show()



y_booking_per_hour = booking_per_hour.values.reshape(-1,1)
X_per_hour = hour_in_day.values.reshape(-1,1)
Linear_Regression_for_campaign(X_per_hour, y_booking_per_hour)
""" 
>>> Linear_Regression_for_campaign(X_per_hour, y_booking_per_hour)
Regressor intercept :  [40.74865441]
Regressor Coefficient :  [[-0.01620318]]
Coefficient of determination: 0.00038114279186785005
Mean Absolute Error: 5.372415339762277
Mean Squared Error: 48.81505402113526
Root Mean Squared Error: 6.986777083973358
"""
Linear_Regression_for_campaign(data_sorted['clickouts'].values.reshape(-1,1), data_sorted['booking'].values.reshape(-1,1))
""" 
Regressor intercept :  [0.12800077]
Regressor Coefficient :  [[-0.01264203]]
Coefficient of determination: 0.0020776542274185683
Mean Absolute Error: 0.1749910839374949
Mean Squared Error: 0.08764412122914908
Root Mean Squared Error: 0.29604749826531057
"""

# There is a negative realtionship between clickouts and booking.
# The coefficient of dteremination is very low to prove it.

data_postive_time = data_sorted[data_sorted['Time_spent_on_session_in_seconds']>0]
y_data_postive_time_booking = data_postive_time.booking.values.reshape(-1,1)
X_data_postive_time_spent = data_postive_time.Time_spent_on_session_in_seconds.values.reshape(-1,1)
Linear_Regression_for_campaign(X_data_postive_time_spent, y_data_postive_time_booking)
"""
Regressor intercept :  [0.05168318]
Regressor Coefficient :  [[0.00020917]]
Coefficient of determination: 0.00197897870191166
Mean Absolute Error: 0.17492384804309905
Mean Squared Error: 0.09347613251584114
Root Mean Squared Error: 0.3057386670276449

"""

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X_data_postive_time_spent, y_data_postive_time_booking, test_size=0.3, random_state=0)
logreg.fit(X_train,y_train )
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))