import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
import warnings
from sklearn.linear_model import LinearRegression

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
    #plt.scatter(X_test, y_test,  color='gray')
    #plt.plot(X_test, y_pred, color='red', linewidth=2)
    #plt.show()

data = pd.read_csv(r'D:\C++\PYTHON\ml\trivago_market_intelligence\1921 - Data Analyst - MINT\marketing_campaigns.csv',sep=';')
data2 = pd.read_csv(r'D:\C++\PYTHON\ml\trivago_market_intelligence\1921 - Data Analyst - MINT\session_data.csv', sep=';')

for df in (data, data2):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data


# There is no missing data

# Market Development campaign wise
camp_a = data[data['Campaign'] == "Aldebaran"]
camp_b = data[data['Campaign'] == "Bartledan"]
camp_c = data[data['Campaign'] == "Cottington"]


# Plotting vistis for differnt campaigns

# Plotting vistis for differnt campaigns, Visits per week for all the campaign
ax = plt.gca()
for df , df_name in zip([camp_a, camp_b,camp_c],('camp_a', 'camp_b', 'camp_c') ):
    df.plot(kind='line',x='Week',y='Visits',ax=ax, label =df_name, title = 'Visits vs Week')
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Visits")


plt.savefig('Visits vs Week.png')
plt.show()

# Plotting Revenue for differnt campaigns, Revenue per week for all the campaigns per week
ax = plt.gca()
for df , df_name in zip([camp_a, camp_b,camp_c],('camp_a', 'camp_b', 'camp_c') ):
    df.plot(kind='line',x='Week',y='Revenue',ax=ax, label =df_name, title = 'Revenue vs Week')
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Revenue")

plt.savefig('Revenue vs Week.png')
plt.show()

# Plotting Costs for differnt campaigns
ax = plt.gca()
for df , df_name in zip([camp_a, camp_b,camp_c],('camp_a', 'camp_b', 'camp_c') ):
    df.plot(kind='line',x='Week',y='Cost',ax=ax, label =df_name, title = 'Cost vs Week')
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Costs")

plt.savefig('Cost vs Week.png')
plt.show()


# Dropping and reseting index, 
for s in (camp_a,camp_b,camp_c):
    s.reset_index(inplace=True, drop=True)


#market_dev_total = camp_a.iloc[:,2:5].add((camp_b.iloc[:,2:5].add(camp_c.iloc[:,2:5], fill_value=0)), fill_value=0)

# Total market development , I am trying to see the overall market impact as well. 
# So I am calculating the total cost, revenue and visits for the market 

c = camp_a['Cost']+camp_b['Cost']+camp_c['Cost']
r = camp_a['Revenue']+camp_b['Revenue']+camp_c['Revenue']
v = camp_a['Visits']+camp_b['Visits']+camp_c['Visits']

total = pd.DataFrame({'week': range(1,32),'visit':v, 'cost':c,'revenue':r})
total.drop(total.tail(1).index,inplace=True) 

ax1 = plt.gca()
for column in ('visit','cost','revenue'):
    total.plot(kind='line',x='week',y=column,ax=ax1,label = column,title = "Total Market Development")

plt.savefig('Total_Market_Revenue.png')
plt.show()

############################################
# question - 2 
#  development of the quality of traffic, e.g. in terms of revenue per visitor
camp_c.drop(camp_c.tail(1).index,inplace=True) 

devlopment_quality_total = total['revenue']/total['visit']
dq_camp_a = camp_a['Revenue']/camp_a['Visits']
dq_camp_b = camp_b['Revenue']/camp_b['Visits']
dq_camp_c = camp_c['Revenue']/camp_c['Visits']

"""# dropping and reseting index
for s in (dq_camp_a,dq_camp_b,dq_camp_c):
    s.reset_index(inplace=True, drop=True)
"""

dq = pd.DataFrame({'week':range(1,31), 'dq_total':devlopment_quality_total,'dq_a':dq_camp_a, 'dq_b':dq_camp_b, 'dq_c':dq_camp_c})

ax2= plt.gca()
for column, label in zip(('dq_total','dq_a','dq_b','dq_c'), ('Total', 'A', 'B','C')):
    dq.plot(kind='line',x='week',y=column, ax=ax2, label = label, title = 'Development Quality')
    ax2.set_xlabel("Weeks")
    ax2.set_ylabel("Revenue Per Visit")

plt.savefig('Developement Quality.png')
plt.show()

# development quality : cost per visit

cost_per_visit_total = total['cost']/total['visit']
cost_per_visit_camp_a = camp_a['Cost']/camp_a['Visits']
cost_per_visit_camp_b = camp_b['Cost']/camp_b['Visits']
cost_per_visit_camp_c = camp_c['Cost']/camp_c['Visits']

"""
# dropping and reseting index
for s in (cost_per_visit_camp_a,cost_per_visit_camp_b,cost_per_visit_camp_c):
    s.reset_index(inplace=True, drop=True)
"""

cost_per_visit = pd.DataFrame({'week':range(1,31), 'cpv_total':cost_per_visit_total,
'cpv_a':cost_per_visit_camp_a, 'cpv_b':cost_per_visit_camp_b, 'cpv_c':cost_per_visit_camp_c})

ax3= plt.gca()
for column, label in zip(('cpv_total','cpv_a','cpv_b','cpv_c'), ('Total', 'A', 'B','C')):
    cost_per_visit.plot(kind='line',x='week',y=column, ax=ax3, label = label, title = 'Cost Per Visit')
    ax3.set_xlabel("Weeks")
    ax3.set_ylabel("Costs per Visit")

plt.savefig('Cost Per Visit.png')
plt.show()

############## 
# Revenue cost ratio , this helps to see how the revenue and cost corelate with each other. 
# Ratio > 1 would imply there is more revenue per cost and this is a favorable situation. The larger the revenue cost ratio, better it is. 
# Ration < 1 would imply that the cost is higher and less revenure is earned. This is not a favorable situation.

rev_cost_total = total['revenue']/total['cost']
rev_cost_camp_a = camp_a['Revenue']/camp_a['Cost']
rev_cost_camp_b = camp_b['Revenue']/camp_b['Cost']
rev_cost_camp_c = camp_c['Revenue']/camp_c['Cost']


rev_cost = pd.DataFrame({'week':range(1,31), 'rev_cost_total':rev_cost_total,
'rev_cost_a':rev_cost_camp_a, 'rev_cost_b':rev_cost_camp_b, 'rev_cost_c':rev_cost_camp_c})

ax4 = plt.gca()
for column, label in zip(('rev_cost_total','rev_cost_a','rev_cost_b','rev_cost_c'), ('Total', 'A', 'B','C')):
    rev_cost.plot(kind='line',x='week',y=column, ax=ax4, label = label, title = 'Revenue Cost Ratio')
    ax4.set_xlabel("Weeks")
    ax4.set_ylabel("Revenue Cost Ratio")

plt.savefig('Revenue Cost Ratio.png')
plt.show()

### Return On Investment 
# Return on Investment is ratio of the net profit and the cost of investment.
# A higher ROI more investment gain ROI is used to evaluate the efficiency of an 
# investment or to compare the efficiencies of several different investments


roi_total = ((total['revenue']-total['cost'])/total['cost'])*100
roi_a = ((camp_a['Revenue']-camp_a['Cost'])/camp_a['Cost'])*100
roi_b = ((camp_b['Revenue']-camp_b['Cost'])/camp_b['Cost'])*100
roi_c = ((camp_c['Revenue']-camp_c['Cost'])/camp_c['Cost'])*100

roi = pd.DataFrame({'week':range(1,31), 'roi_total':roi_total,
'roi_a':roi_a, 'roi_b':roi_b, 'roi_c':roi_c})

ax5 = plt.gca()
for column, label in zip(('roi_total','roi_a','roi_b','roi_c'), ('Total', 'A', 'B','C')):
    roi.plot(kind='line',x='week',y=column, ax=ax5, label = label, title = 'Return on Investment')
    ax5.set_xlabel("Weeks")
    ax5.set_ylabel("Return on Investment")

plt.savefig('Return on Investment.png')
plt.show()

### Return on inversment per visit
# I am calulating g ROI per visit to normalie the ROI for vists across differnt campaigns. 
roi_total_per_visit = ((total['revenue']-total['cost'])/total['cost']*total['visit'])*100
roi_a_per_visit = ((camp_a['Revenue']-camp_a['Cost'])/camp_a['Cost']*camp_a['Visits'])*100
roi_b_per_visit = ((camp_b['Revenue']-camp_b['Cost'])/camp_b['Cost']*camp_b['Visits'])*100
roi_c_per_visit = ((camp_c['Revenue']-camp_c['Cost'])/camp_c['Cost']*camp_c['Visits'])*100

roi_per_visit = pd.DataFrame({'week':range(1,31), 'roi_total_per_visit':roi_total_per_visit,
'roi_a_per_visit':roi_a_per_visit, 'roi_b_per_visit':roi_b_per_visit, 'roi_c_per_visit':roi_c_per_visit})

ax6 = plt.gca()
for column, label in zip(('roi_total_per_visit','roi_a_per_visit','roi_b_per_visit','roi_c_per_visit'), ('Total', 'A', 'B','C')):
    roi_per_visit.plot(kind='line',x='week',y=column, ax=ax6, label = label, title = 'Return on Investment Per Visit')
    ax6.set_xlabel("Weeks")
    ax6.set_ylabel("Return on Investment per Visit")

plt.savefig('Return on Investment per Visit.png')
plt.show()

#################################################################################################

# Fitting Linear Regression to model the revenue on cost

a_revenue = camp_a.Revenue
a_cost = camp_a.Cost

X_a = a_cost.values.reshape(-1,1)
y_a = a_revenue.values.reshape(-1,1)

b_revenue = camp_b.Revenue
b_cost = camp_b.Cost
X_b = b_cost.values.reshape(-1,1)
y_b = b_revenue.values.reshape(-1,1)

c_revenue = camp_c.Revenue
c_cost = camp_c.Cost
X_c = c_cost.values.reshape(-1,1)
y_c = c_revenue.values.reshape(-1,1)


# High coefficient of determination implies that there is high correlation between cost and revenue. 

# Revenue for Campaingn A for 31st week for 250 
Linear_Regression_for_campaign(X_a,y_a)
Linear_Regression_for_campaign(X_b,y_b)
Linear_Regression_for_campaign(X_c,y_c)

"""
>>> Linear_Regression_for_campaign(X_a,y_a)
Regressor intercept :  [-15.81085367]
Regressor Coefficient :  [[1.17228303]]
Coefficient of determination: 0.9841774010454576
Mean Absolute Error: 12.43478579462804
Mean Squared Error: 198.0071101432279
Root Mean Squared Error: 14.071499925140458

>>> Linear_Regression_for_campaign(X_b,y_b)
Regressor intercept :  [-1.30926405]
Regressor Coefficient :  [[0.90692254]]
Coefficient of determination: 0.9931492952273674
Mean Absolute Error: 8.79461708872244
Mean Squared Error: 90.887604234626
Root Mean Squared Error: 9.533499055154198

>>> Linear_Regression_for_campaign(X_c,y_c)
Regressor intercept :  [134.30829701]
Regressor Coefficient :  [[0.66030385]]
Coefficient of determination: 0.9510064819457942
Mean Absolute Error: 6.011922361326602
Mean Squared Error: 44.76744706124759
Root Mean Squared Error: 6.6908480076330825
    """

# for 250 addition euros, revenue for each campaign 

additional_revenue_a = -15.81085367 + 250*1.17228303
additional_revenue_a
additional_revenue_b = -1.30926405 + 250*0.90692254
additional_revenue_b
additional_revenue_c = 134.30829701 + 0.66030385
additional_revenue_c

""" 
>>> additional_revenue_a = -15.81085367 + 250*1.17228303
>>> additional_revenue_a
277.25990383
>>> additional_revenue_b = -1.30926405 + 250*0.90692254
>>> additional_revenue_b
225.42137095
>>> additional_revenue_c = 134.30829701 + 0.66030385
>>> additional_revenue_c
134.96860085999998        
        """

# The revenue is maximum for Campaign A as expected 
# FOr additional 250 euros the revenue generated for each campaign is listed above. It is evident that the revenue is hightes for
# campaign A. And the Return of investment and return of investment per visit is also has a positive growing trend. Hence I would
# highly recommed to invest more in Campaign A. 

# The return on investment is increasing with week for campaign A. And seem to be decreasing. 
# I am using Linear Regression to quantify the growth of the ROI on week.
# So X = week from 1 - 30 and the target variable is the reurn on investment

X_roi = camp_a.Week.values.reshape(-1,1)
y_roi_a = roi_a.values.reshape(-1,1)
y_roi_b = roi_b.values.reshape(-1,1)
y_roi_c = roi_c.values.reshape(-1,1)

Linear_Regression_for_campaign(X_roi, y_roi_a)
Linear_Regression_for_campaign(X_roi, y_roi_b)
Linear_Regression_for_campaign(X_roi, y_roi_c)

""" 
>>> Linear_Regression_for_campaign(X_roi, y_roi_a)
Regressor intercept :  [-39.66292514]
Regressor Coefficient :  [[1.97808412]]
Coefficient of determination: 0.9750769463739978
Mean Absolute Error: 1.6921513265973942
Mean Squared Error: 7.039562745938856
Root Mean Squared Error: 2.6532174328424074

>>> Linear_Regression_for_campaign(X_roi, y_roi_b)
Regressor intercept :  [-10.58194813]
Regressor Coefficient :  [[0.04945747]]
Coefficient of determination: 0.02269246617010634
Mean Absolute Error: 3.241452120937344
Mean Squared Error: 10.807924671180237
Root Mean Squared Error: 3.287540824260626

>>> Linear_Regression_for_campaign(X_roi, y_roi_c)
Regressor intercept :  [19.12978328]
Regressor Coefficient :  [[-0.90819857]]
Coefficient of determination: 0.8743393851764616
Mean Absolute Error: 1.813422401672149
Mean Squared Error: 3.7128994107154707
Root Mean Squared Error: 1.9268885309522892
"""

# The slope of ROI vs Week for Campaign A, B, C is 
# A : 1.97808412
# B : 0.04945747
# C : -0.90819857
# A has the highest positive increasing return on investment. 
# Hence A would be the best campaign to invest more money in.
