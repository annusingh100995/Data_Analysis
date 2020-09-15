# https://blog.cambridgespark.com/tutorial-practical-introduction-to-recommender-systems-dbe22848392b
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/

""" Implementing a simple recommender system. This analysis is only to find the movies which are similarto certain other
    in a list of movies."""

# importing libraries

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# seaborn is a statistical data visualization library
# path to download the data 

path = 'https://media.geeksforgeeks.org/wp-content/uploads/file.tsv'

# df to store the data
""" The data frame consists of movie rating data, with 4 coulmns,
    user id, item id, rating and time stamp
    Hence we name the column of the dataframe while reading it. 
    These are the rating given by different users to differnet movies. 
    Every user and movie has a unique id.
    lenght of the df = 100003 """

column_names = ['user_id', 'item_id', 'rating', 'timestamp'] 

#Reading the data 
df = pd.read_csv(path, sep='\t', names=column_names)
df.head()

# This the list of the movies and their unique ids.
movie_titles = pd.read_csv('https://media.geeksforgeeks.org/wp-content/uploads/Movie_Id_Titles.csv') 
movie_titles.head() 

# merging the two data frames, such that movie names are attached to the respective movie ids. 

data = pd.merge(df,movie_titles, on='item_id')

# this is just for exploration
# This groups the data by titles, then finds the mean
data.groupby('title')['rating'].mean().sort_values(ascending=False).head() 

# this counts the number of rating each movies is given
data.groupby('title')['rating'].count().sort_values(ascending=False).head() 


# Here, the means of rating and the frequency of rating for each movie is
# stored in a dataframe rating
ratings = pd.DataFrame(data.groupby('title')['rating'].mean())  
  
ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count()) 
  
ratings.head()

plt.figure(figsize=(10,4))

ratings['num of ratings'].hist(bins=100)
plt.title('The number of ratings ')
plt.show()

# Plotting the ratings
plt.figure(figsize =(10, 4)) 
ratings['rating'].hist(bins = 70)
plt.title('Ratings')
plt.show() 
# it can be seen that 3 is the average rating

# pivot tabel creates a multi index object data frmae
"""so, here with the index = user_id, ie for each user in the df data 
    pivot tabel creats a df with rating gievn by that user for each movie
    If no rating is given NaN is applied
"""
moviemat = data.pivot_table(index ='user_id',columns ='title', values ='rating') 
  
moviemat.head() 
  
ratings.sort_values('num of ratings', ascending = False).head(10) 

# Rating given by each user to starwars and liarliar
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)'] 

#Compute pairwise correlation between rows or columns of DataFrame with rows or columns of Series or DataFrame.
similar_to_starwars = moviemat.corrwith(starwars_user_ratings) 
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings) 
  
# finding the corraltion of each  movie with statwars
corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation']) 
corr_starwars.dropna(inplace = True) 
  
corr_starwars.head() 

010 u4-
corr_starwars.sort_values('Correlation', ascending = False).head(10) 
corr_starwars = corr_starwars.join(ratings['num of ratings']) 
  
corr_starwars.head() 
  
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending = False).head() 




import numpy as np
import pandas as pd
import urllib
import io
import zipfile

tmpfile = urllib.request.urlopen('https://www.librec.net/datasets/filmtrust.zip')

tmpfile = zipfile.ZipFile(io.BytesIO(tmpfile.read()))

dataset = pd.read_table(io.BytesIO(tmpfile.read('ratings.txt'),
    sep='',names = ['uid', 'iid', 'rating']))

tmpfile.close()
dataset.head()


#https://medium.com/hacktive-devs/recommender-system-made-easy-with-scikit-surprise-569cbb689824
#https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split


#data = pd.read_csv(r'D:\C++\PYTHON\ml\', encoding='utf-8')
#https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101
articles_df = pd.read_csv(r'D:\C++\PYTHON\ml\articles-sharing-reading-from-cit-deskdrop\shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']

interactions_df = pd.read_csv(r'D:\C++\PYTHON\ml\articles-sharing-reading-from-cit-deskdrop\users_interactions.csv')


# Giving strength to each type of interaction
event_type_strength = {'VIEW ': 1.0, 'LIKE': 2.0, 'BOOKMARK':2.5, 'FOLLOW':3.0, 'COMMENT CREATED' : 4.0 }

# Adding the interaction sterength to the interaction data
# Create a column called eventStrength and add the strenght accorging to the eventType
interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()

# printing the length of the users. A single user can have interaction with many items and also many interactions with a single item.
print('# users: %d' % len(users_interactions_count_df))

# selecting the users that has alteat 5 ineractions
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]

# Number of users with enough interactions
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

# these are the total number of interactions
print('# of interactions: %d' % len(interactions_df))

# merging the two columns together on personID being the common column between the two
# Here i am merging the the df "users_with_enough_interactions_df" and "interactions_df", so all the interactions a user woth more than
# 5 interaction are collected.
# so, interactions_from_selected_users_df are the users with alleast 5 interactions and their all interactions
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')

def smooth_user_preference(x):
    return math.log(1+x, 2)

"""Okay, so one user can interact with multiple items and with one item multiple times. 
    Now, for each user the item with which it interact is grouped together. Then the sum of each interaction for each item is calculated
    and smoothened using the smooth_user_preference
    reset_index will keep the index for each item if not set , the user id is not set for each item.
    -9223121837663643404  -8949113594875411859    1.000000
                      -8377626164558006982    1.000000
                      -8208801367848627943    1.000000
                      -8187220755213888616    1.000000
                      -7423191370472335463    3.169925
                      
                      
      with reset index                
    0  -9223121837663643404 -8949113594875411859       1.000000
      -9223121837663643404 -8377626164558006982       1.000000
    2  -9223121837663643404 -8208801367848627943       1.000000
    3  -9223121837663643404 -8187220755213888616       1.000000
    4  -9223121837663643404 -7423191370472335463       3.169925"""

interactions_full_df = interactions_from_selected_users_df.groupby(['personId', 'contentId'])['eventStrength'].sum().apply(smooth_user_preference).reset_index()

#Splitting the data into test and train sets

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,stratify=interactions_full_df['personId'], 
    test_size=0.20,random_state=42)

# printing the number of interactions in the training and the test dataset
print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

