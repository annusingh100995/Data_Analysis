import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
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


def features_vs_time(dataframe,feature,plt_title,plt_ylabel, save_as, save_as_pie):
    # Dividing the data into 24 groups. One group for each hour of the day.
    df = dataframe.groupby(dataframe['sess_start_time'].map(lambda x: x.hour))
    count = dataframe.groupby(dataframe['sess_start_time'].map(lambda x: x.hour))[feature].value_counts()
    a = pd.DataFrame(count)
    # List for counts where booking == 1 for each hour of the day.
    count_per_hour_1 =[]
    for i in range(0,24):
        count_per_hour_1.append(a[feature][i][1])
    hour_in_day = pd.Series(range(0,24))
    feature_per_hour = pd.Series(count_per_hour_1)
    # Pie chart
    (feature_per_hour).plot.pie(labels= range(0,24), autopct='%.2f', fontsize=10, figsize=(6, 6))
    plt.title(plt_title)
    plt.savefig(save_as_pie)
    plt.show()
    plt.plot(hour_in_day , feature_per_hour)
    plt.title(plt_title)
    plt.xlabel('Time (0 HRS - 23 HRS)')
    plt.ylabel(plt_ylabel)
    plt.xticks(hour_in_day)
    plt.yticks(feature_per_hour)
    plt.savefig(save_as)
    plt.show()
    y_feature_per_hour = feature_per_hour.values.reshape(-1,1)
    X_per_hour = hour_in_day.values.reshape(-1,1)
    Linear_Regression_for_campaign(X_per_hour, y_feature_per_hour)

features_vs_time(data_sorted,'clickouts', 'Clickouts vs Hour', 'Clickouts per Hour', 'Clickouts vs Hours (0-23)', 'Clickouts vs Time pie')
"""
>>> features_vs_time(data_sorted,'clickouts', 'Clickouts vs Hour', 'Clickouts per Hour', 'Clickouts vs Hours (0-23)', 'Clickouts vs Time pie')
Regressor intercept :  [51.027024]
Regressor Coefficient :  [[0.42958062]]
Coefficient of determination: 0.13718934382029568
Mean Absolute Error: 3.148553487328999
Mean Squared Error: 13.07553535662521
Root Mean Squared Error: 3.6160109729680316
"""