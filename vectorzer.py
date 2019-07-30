from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(_file_)
stop = pickle.load(open(os.path.join(cur_dir,'pkl_obkects','stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('?::|;|=)?:-)?(:\)|\(|D|P)',text.lower())
        +''.join(emoticons).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features 2**21, preporcessor=None, tokenizer=tokenizer)