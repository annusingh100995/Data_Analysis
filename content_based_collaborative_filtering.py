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


corr_starwars.sort_values('Correlation', ascending = False).head(10) 
corr_starwars = corr_starwars.join(ratings['num of ratings']) 
  
corr_starwars.head() 
  
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending = False).head() 
