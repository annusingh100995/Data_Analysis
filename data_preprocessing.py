import pandas as pd
from io import StringIO

# Making a sample incomplete data
csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''

# converting the file into a dataframe for data manipulation
df = pd.read_csv(StringIO(csv_data))

#sum: return the number of missing values per column
df.isnull().sum()

#Elimiating the samples, , drops the row(axis =1) column(axis = 0) where the data is missing
df.dropna(axis=0)

df.dropna(axis=1)

#or drop all , drops all rows and columns where there are missing values

df.dropna(how='all')

# drops where Nan appers in specific rows, here C
df.dropna(subset=['C'])

"""

# IMPUTING MISSING VALUES   
from sklearn.impute import SimpleImputer
imr = Imputer(missing_values='NaN', strategy='mean',axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

# some library issues
"""

# Nominal and Ordinal(a comparing order can be defined) Features

import pandas as pd
df = pd.DataFrame([
    ['green','M',10.1,'class1'],
    ['red','L',13.5,'classs2'],
    ['blue','XL',15.4,'class1']])
df.columns = ['color','size','price','classlabel']

#MApping ordinal features
# Defining a size mapping ruel for nominal reature size, a mapping size dictionary
size_mapping ={'XL':3,'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
df

# Reversing the mapping back to the original feature
# Defining the inverse of the mapping size dictionary
inv_size_mapping = {v: k for k , v in size_mapping.items()}
df['size'].map(inv_size_mapping)

# Encoding class label, useful when there is no need for the label to be in any order, just 
# distinguished integral labels are good
# Creating a mapping dictionary
import numpy as np
class_mapping = {label:idx for idx ,label in enumerate(np.unique(df['classlabel']))}

# using the dictionary to transform the label into integars\\
# The classlabel column of df is mapped to the intergars using the clas mapping dictionary
df['classlabel'] =df['classlabel'].map(class_mapping)

# reversing the key-values

inv_class_mapping = {v:k for k , v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

 # Lable Encoder form the sklearn 
 from sklearn.preprocessing import LabelEncoder
 class_le = LabelEncoder()
 y = class_le.fit_transform(df['classlabel'].values)
 y
 class_le.inverse_transform(y)

""" One Hot Encoding
    the idea here is to create a dummy feature to encode the original feature
        This is done to ensure that there is no unnecessary comparison happening after feature conversion
"""
X = df[['color','size','price']].values
color_le = LabelEncoder()
X[:,0] =color_le.fit_transform(X[:,0])
X

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
#ohe.fit_transform(X).toarray() # some errror


# more convient method to create the dummy features
pd.get_dummies(df[['price','color','size']])

# importing the wine data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

df.columns = ['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
    'Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
    'Proline']


# Partioning the data and creating the traning and the test data 

from sklearn.model_selection import train_test_split
X,y = df.iloc[:,1:].values, df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
# stratify ensures that both training and test data set have same class proportion

# Scaling the features, min max methods
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Scaling via standardisation

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

# Tacking ovefitting by regularisation , using L1 regularisation 

from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')

# lambda is the regularisation parameter,
lr = LogisticRegression(penalty='l1',C=1.0)
lr.fit(X_train_std, y_train)

print('Training accuracy:' , lr.score(X_train_std, y_train))
print('Test accuracy:' , lr.score(X_test_std, y_test))

# to find the intercept
lr.intercept_

# Plotting a curve,  varying the regularisation strength and plot the regularisaion path 
# The weight coefficient for different features for differnt regularisation strength

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue','green', 'red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
weights , param = [], []

for c in np.arange(-4. , 6.):
    lr = LogisticRegression(penalty='l1', C= 10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    param.append(10**c)

weights = np.array(weights)

for column,color in zip(range(weights.shape[1]), colors):
    plt.plot(param, weights[:,column], label=df.columns[column+1], color=color)

plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficeient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='lower left')
ax.legend(loc='upper center',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show()

""" Assessing the important feature with RANDOM FOREST """
from sklearn.ensemble import RandomForestClassifier

feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train, y_train)
# This is to get hte inportance of each feature of the data, values between 0 and 1
importances = forest.feature_importances_

# Sorting the indices of the features 
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" %(f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('FEATURE IMPORTANCE')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')

plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

""" Selecting the most important feature"""

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of asmples that meet this criterion:', X_selected.shape[0])



for f in range(X_selected[1]):
    print("%2d) %-*s %f" %(f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

for f in range(X_selected.shape[1]):
    print(" %2d) -*s %f" %(f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

