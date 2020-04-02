""" Sentiment Analysis for Movie Review Data"""

# PyPrind:  Python Progress Indicator: is used to visualize the progress and estimate the time unitl completion 

import pyprind
import pandas as pd
import os

basepath = r'D:\C++\PYTHON\ml\aclImdb'

labels = {'pos':1 , 'neg':0}

pbar = pyprind.ProgBar(50000)

"""  Here, the data is combined into a single data frame. 
    Through the for loop, the reviews in the test and the training directories is fetched and mergerd into a single dataframe"""

""" So test has two sub directories : pos and neg., so the following loop,goes to the train folder, thenthe pos and neg subfolder of the
    train folder and then appends the reviews to the df and then repeats the same for the test folder"""
df = pd.DataFrame()

for s in ('test','train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath,s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()



# Since the class labels in the assembles data are sorted,the data is shuffled,this will help when the data will be split 
# into training and test data

import numpy as np 
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data_csv', index=False, encoding='utf-8')

# the data frame is then converted into csv file

df = pd.read_csv('movie_data_csv', encoding='utf-8')
df.head(10)
df.columns = ['review', 'sentiment']
#Naming the columns, just in case the names weren't there


# Use this to load the file 
# df = pd.read_csv(r'D:\C++\PYTHON\ml\movie_data_csv', encoding='utf-8')


# Cleaning the data
""" Since the data have unnecssary stuf like markup quotes. So data cleaning is required to remove this
    and also collect emoticons as they would be benefical for sentiment analysis."""

import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', '', text.lower()) + ''.join(emoticons).replace('-',''))
    return text 


# Applying the cleaning function to all the dataset
df['review'] = df['review'].apply(preprocessor)


# Word Stemming : finding the root word 
def tokenizer(text):
    return text.split()

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
# NLTK : Natural Language Tool Kit

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

################################################################################

# Training a logistic  regression model for document analysis

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'review'].values

X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'review'].values 

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Term frquency- inverse document frequency

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range':[(1,1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer,tokenizer_porter],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]},
                

                {'vect__ngram_range': [(1,1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf': [False],
                'vect__norm':[None],
                'clf__penalty':['l1','l2'],
                'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=1,solver='lbfgs'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=1)

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s' %gs_lr_tfidf.best_params_)
print('Best score: %s' %gs_lr_tfidf.best_score_)
