"""Topic Medelling is the broad task of assigning topic to inlabelled text.
    Gereally used in the categoriazation of large text documents
    LDA: Latent Dirichlet Allocation is a popular technique for topic modelling
    
        LDA is a generative probabilistic model that tries to find the groups of words that apperar frequently across
        differnet documents. 
        These frequently occuring words represent the topics."""

import pandas as pd

# Importing the previously manipulated movie data csv file
df = pd.read_csv(r'D:\C++\PYTHON\ml\movie_data_csv', encoding='utf-8')
#Naming the columns just in case
df.columns = ['review', 'sentiment']

#importing the stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Importing the count
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000 )
# mac_df = 0.1: the maximum document frequency is set to 10%, this is done to ignore words that
# occur too frequently in the documents
# max_features: number of words to be condsidered is set to 5000, to limit the dimesionality of the dataset
X = count.fit_transform(df['review'].values)

# Using LDA to find 10 topics

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=10, random_state=123, learning_method='batch')
# Leaning method is a method to do the estimations based on the available training data 
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
# use this , adaptable to my pc version 

X_topics = lda.fit_transform(X)

n_top_words = 5
features_names = count.get_feature_names()
for topic_idx , topic in enumerate(lda.components_):
    print("Topic %d: " %(topic_idx+1))
    print(" ".join([features_names[i] for i in topic.argsort() [:-n_top_words-1:-1]]))