# Loading the data and importing the required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv(r'D:\C++\PYTHON\trivago_case\train_set.csv')
test = pd.read_csv(r'D:\C++\PYTHON\trivago_case\test_set.csv')

#train_backup = train
#test_backup = test

#Column name for inspection
train.columns
"""Index(['hotel_id', 'city_id', 'content_score', 'n_images',
       'distance_to_center', 'avg_rating', 'stars', 'n_reviews', 'avg_rank',
       'avg_price', 'avg_saving_percent', 'n_clicks'],
      dtype='object')"""

test.columns
""" Index(['hotel_id', 'city_id', 'content_score', 'n_images',
       'distance_to_center', 'avg_rating', 'stars', 'n_reviews', 'avg_rank',
       'avg_price', 'avg_saving_percent'],
      dtype='object') """

print("The train data size before dropping Id feature is : {} ".format(train.shape))
#The train data size before dropping Id feature is : (396487, 12) There are 396487 samples and12 features

print("The test data size before dropping Id feature is : {} ".format(test.shape))
#The test data size before dropping Id feature is : (132162, 11)

# Now drop the 'hotel_id' column since it's unnecessary for the prediction process.
train.drop("hotel_id", axis = 1, inplace = True)
test.drop("hotel_id", axis = 1, inplace = True)


# Check data size after dropping the 'hotel_id' variable
print("\nThe train data size after dropping hotel_id feature is : {} ".format(train.shape)) 
# The train data size after dropping hotel_id feature is : (396487, 11)

print("The test data size after dropping hotel_id feature is : {} ".format(test.shape))
# The test data size after dropping hotel_id feature is : (132162, 10)


# 2. Analysing the test variable that is the n_clicks
 
# Getting Description
train['n_clicks'].describe()

""" 
count    396487.000000
mean         13.781980
std         123.572896
min           0.000000
25%           0.000000
50%           0.000000
75%           2.000000
max       13742.000000
Name: n_clicks, dtype: float64      """

# Plotting the n_clicks just to get a view of how the data looks like
plt.plot(train['n_clicks'])
plt.show()

# Plot Histogram
sns.distplot(train['n_clicks'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['n_clicks'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('n_clicks distribution')

fig = plt.figure()
res = stats.probplot(train['n_clicks'], plot=plt)
plt.show()

print("Skewness: %f" % train['n_clicks'].skew())
print("Kurtosis: %f" % train['n_clicks'].kurt())


# 3. MULTIVARICATE ANALYSIS
""" Here, I try to look at the data. Its distribution and corelation. 
I try to find the festures that may be very important for predicting the number of clicks for the hotels.
I also try to see how each feature is related to the n_clicks. 
I plot the relation between each feature and the number of clicks"""

# Checking the numeric and categorical data
train.select_dtypes(include=['object']).columns
train.select_dtypes(include=['int64','float64']).columns

# the training data has all numeric data which reduces few steps of converting any categorical data into numeric data.
cat = len(train.select_dtypes(include=['object']).columns)
num = len(train.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+',
      num, 'numerical', '=', cat+num, 'features')

# Total Features:  0 categorical + 11 numerical = 11 features

# Drawing a correlation matrix heat map , to find out if there is any quite obvious correaltion between the features
# and the n_clicks

corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# Top 10 Heatmap
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'n_clicks')['n_clicks'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr

"""    Most Correlated Features
0                  n_clicks
1                 n_reviews
2        avg_saving_percent
3                     stars
4             content_score
5                 avg_price
6                avg_rating
7                  n_images
8        distance_to_center
9                   city_id
10                 avg_rank 

The most correlated features to the n_clicks rankes in order
Side note: n_clicks will be most correlated to itself . """

# feature n_reviews vs n_clicks
train['n_reviews'].describe()

var = 'n_reviews'
data = pd.concat([train['n_clicks'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="n_clicks", data=data)
fig.axis(ymin=0, ymax=300000);

plt.scatter(train['n_reviews'], train['n_clicks'], edgecolors='r')
plt.xlabel('n_reviews')
plt.ylabel('n_clicks')
plt.title('n_reviews vs n_clicks')
plt.show()

# avg_saving_percent vs n_clicks
sns.jointplot(x=train['avg_saving_percent'], y=train['n_clicks'], kind='reg')

# scatter plot
plt.scatter(train['avg_saving_percent'], train['n_clicks'], edgecolors='r')
plt.xlabel('avg_saving_percent')
plt.ylabel('n_clicks')
plt.title('avg_saving_percent vs n_clicks')
plt.show()

""" STAR VS N_CLICKS"""
# stars vs n_clicks
sns.jointplot(x=train['stars'], y=train['n_clicks'], kind='reg')
plt.show()
# scatter plot
plt.scatter(train['stars'], train['n_clicks'], edgecolors='r')
plt.xlabel('stars')
plt.ylabel('n_clicks')
plt.title('stars vs n_clicks')
plt.show()

# quite visible that more number of stars mean more numer of clicks on that hotel

""" CONTENT_SCORE VS N_CLICKS """
#  content_score vs n_clicks
sns.jointplot(x=train['content_score'], y=train['n_clicks'], kind='reg')
plt.show()
# scatter plot
plt.scatter(train['content_score'], train['n_clicks'], edgecolors='r')
plt.xlabel('content_score')
plt.ylabel('n_clicks')
plt.title('content_score vs n_clicks')
plt.show()

# from the scatter plot: higher content_score attracts more number of clicks 

""" AVERAGE_PRICE VS N_CLICKS"""
sns.jointplot(x=train['avg_price'], y=train['n_clicks'], kind='reg')
plt.show()
# scatter plot
plt.scatter(train['avg_price'], train['n_clicks'], edgecolors='r')
plt.xlabel('avg_price')
plt.ylabel('n_clicks')
plt.title('avg_price vs n_clicks')
plt.show()

# Quite obvious, lower the average price mean more number of clicks

""" AVERAGE RATING VS N_CLICKS """
sns.jointplot(x=train['avg_rating'], y=train['n_clicks'], kind='reg')
plt.show()
# scatter plot
plt.scatter(train['avg_rating'], train['n_clicks'], edgecolors='r')
plt.xlabel('avg_rating')
plt.ylabel('n_clicks')
plt.title('avg_rating vs n_clicks')
plt.show()
# Higher rating attracts more number of clicks

"""  NUMBER OF IMAGES VS N_CLICKS """
sns.jointplot(x=train['n_images'], y=train['n_clicks'], kind='reg')
plt.title('n_images vs n_clicks')
plt.show()
# scatter plot
plt.scatter(train['n_images'], train['n_clicks'], edgecolors='r')
plt.xlabel('n_images')
plt.ylabel('n_clicks')
plt.title('n_images vs n_clicks')
plt.show()
# WEIRD less number of images have more clicks

""" DISTANCE TO CENTRE VS N_CLICKS"""
sns.jointplot(x=train['distance_to_center'], y=train['n_clicks'], kind='reg')
plt.title('distance_to_center vs n_clicks')
plt.show()
# scatter plot
plt.scatter(train['distance_to_center'], train['n_clicks'], edgecolors='r')
plt.xlabel('distance_to_center')
plt.ylabel('n_clicks')
plt.title('distance_to_center vs n_clicks')
plt.show()
# LESS distance to centre attracts more number of clicks, very obvious

""" CITY ID VS N_CLICKS """
sns.jointplot(x=train['city_id'], y=train['n_clicks'], kind='reg')
plt.title('city_id vs n_clicks')
plt.show()
# scatter plot
plt.scatter(train['city_id'], train['n_clicks'], edgecolors='r')
plt.xlabel('city_id')
plt.ylabel('n_clicks')
plt.title('city_id vs n_clicks')
plt.show()
# certain cities have more number of clicks than other, may be these are major tourist attractions

"""  AGERAGE RANK NS N_CLICK"""
sns.jointplot(x=train['avg_rank'], y=train['n_clicks'], kind='reg')
plt.title('avg_rank vs n_clicks')
plt.show()
# scatter plot
plt.scatter(train['avg_rank'], train['n_clicks'], edgecolors='r')
plt.xlabel('avg_rank')
plt.ylabel('n_clicks')
plt.title('avg_rank vs n_clicks')
plt.show()
# very intersting, low ranking associated with more number of clicks
          
list = ['n_reviews','avg_saving_percent','stars','content_score','avg_price','avg_rating','n_images','distance_to_center',
    'city_id','avg_rank'] 
   
for i, feature in zip(range(1,len(list)),list):
    plt.subplot(2,5,i)
    plt.scatter(train[feature], train['n_clicks'], edgecolors='r')
    plt.xlabel(feature)
    plt.ylabel('n_clicks')
    plt.title(feature)
plt.show()

# DATA CLEANING AND IMPUTING MISSING VALUES

""" It is important to clean the data before analysing it. There can be missing values that can either be imputed or removed.
    Depending upon the data we need to seelct the best way to clean the data.
    We need to check wether the missing values are random or there is some pattern in the missing values.
    We also don't want to remove all the missing data as it  might remove revelant information
    Hence, this is a crucial step and need to be heed properly."""

# number of rows in train and test data set
ntrain = train.shape[0]
ntest = test.shape[0]

# number of clicks of the training dataset
y_train = train.n_clicks.values

# train and test data combined
all_data = pd.concat((train, test)).reset_index(drop=True)
# the n_clicks column removed
all_data.drop(['n_clicks'], axis=1, inplace=True)

print("Train data size is : {}".format(train.shape))
print("Test data size is : {}".format(test.shape))
print("Combined dataset size is : {}".format(all_data.shape))

# Finding the missing data 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

"""all_data_na
avg_price              0.043129
avg_rank               0.000000
avg_rating            27.812027
avg_saving_percent     0.043129
city_id                0.125414
content_score          0.125414
distance_to_center     0.130900
hotel_id               0.000000
n_images               0.125792
n_reviews              0.130900
stars                  0.139412    

    aug_price and hotel_id have zero missing values while the avg_rating is missing the most 27 % is missing """

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data

""" missing_data
                    Missing Ratio
avg_rating              27.812027
stars                    0.139412
n_reviews                0.130900
distance_to_center       0.130900
n_images                 0.125792
content_score            0.125414
city_id                  0.125414
avg_saving_percent       0.043129
avg_price                0.043129   """


# Plotting the statistics of missing data.

f, ax = plt.subplots(figsize=(15, 10))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=10)
plt.ylabel('Percent of missing values', fontsize=10)
plt.title('Percent missing data by feature', fontsize=10)
plt.show()

train['avg_rating'].describe()
"""
count    286089.000000
mean         81.616525
std           6.025140
min          34.000000
25%          79.000000
50%          83.000000
75%          86.000000
max          97.000000
Name: avg_rating, dtype: float64    
    75% of the avg_rating is below 86, mean of the average rating is ~82. I think using median to impute the avg_rating should be a good call
    """"

""" I'm imputing the features one by one , withouth using sklearn imputer. 
    For avg_rating : 27% missing values, I'm imputing the missing values using the median
    For stars, n_reviews, distance_to_center, n_images, content_score, avg_saving_percent the missing values are ~1% and I'm 
        using mean values to replace the missing values
    For city_id:  1.2% values are missing, I am using mode to impute the missing values. 
        As I think the majority of the hotel to be in the major tourist attraction. So the mode gives the is id of the most
        busy city. """

# imptuing the data
all_data['city_id'] = all_data['city_id'].fillna(all_data['city_id'].mode()[0])
for col in ('n_reviews','avg_saving_percent','stars','content_score','avg_price','n_images','distance_to_center','avg_rank'):
    all_data[col] = all_data[col].mode()[0]
all_data['avg_rating'] = all_data['avg_rating'].fillna(all_data['avg_rating'].mean()[0])
all_data["avg_rating"] = all_data["avg_rating"].transform(lambda x: x.fillna(x.median()))

# Checking in missing data is there
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head() 
# no missing data

# FEATURE TRANDFORAMTION
""" The features are most of the time in different range. For example the avrage ratind and the distance.
    hence for propr analyis, it is important to normalise the data to bring it into comparable scale."""

names = train.columns
# Create the Scaler object
scaler = StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(train)
scaled_df = pd.DataFrame(scaled_df, columns=names)

list = ['n_reviews','avg_saving_percent','stars','content_score','avg_price','avg_rating','n_images','distance_to_center',
    'city_id','avg_rank'] 
   
for i, feature in zip(range(1,len(list)),list):
    plt.subplot(2,5,i)
    sns.kdeplot(train[feature])
    plt.title(feature)
plt.show()

for i, feature in zip(range(1,len(list)),list):
    plt.subplot(2,5,i)
    sns.kdeplot(scaled_df[feature])
    plt.title(feature)
plt.show()

from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler

mm_scaler = MinMaxScaler()
nor_scaler = Normalizer()
rob_scaler = RobustScaler()

mm_scaled_df = mm_scaler.fit_transform(train)
mm_scaled_df = pd.DataFrame(mm_scaled_df, columns=names)

nor_scaled_df = scaler.fit_transform(train)
nor_scaled_df = pd.DataFrame(nor_scaled_df, columns=names)

rob_scaled_df = scaler.fit_transform(train)
rob_scaled_df = pd.DataFrame(rob_scaled_df, columns=names)

for i, feature in zip(range(1,len(list)),list):
    plt.subplot(2,5,i)
    sns.kdeplot(rob_scaled_df[feature])
    plt.title(feature)
plt.show()