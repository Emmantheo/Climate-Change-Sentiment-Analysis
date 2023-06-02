"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.5
"""
from sqlite3 import Timestamp
import pandas as pd
import re
#import os
#from pyspark.sql import DataFrame
import texthero as hero
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
#from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from datetime import date,timedelta,datetime
import snscrape.modules.twitter as sntwitter
import time 







def _clean_tweet(tweet):
    '''
    tweet: String
           Input Data
    tweet: String
           Output Data
           
    func: Removes hashtag symbol in front of a word
          Replace URLs with a space in the message
          Replace ticker symbols with space. The ticker symbols are any stock symbol that starts with $.
          Replace  usernames with space. The usernames are any word that starts with @.
          Replace everything not a letter or apostrophe with space
          Remove single letter words
          filter all the non-alphabetic words, then join them again

    '''
    
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub('\$[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('\@[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('[^a-zA-Z\']', ' ', tweet)
    tweet = re.sub(r'\s+', " ", tweet)
    tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    
    return tweet

def _token_stop_pos(text):

    '''
    Maps the part of speech to words in sentences giving consideration to words that are nouns, verbs, 
    adjectives and adverbs
    '''
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


def lemmatize(pos_data):
        '''
        Performs lemmatization on tokens based on its part of speech tagging 
        '''
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_rew = " "
        for word, pos in pos_data:
            if not pos:
                lemma = word
                lemma_rew = lemma_rew + " " + lemma
            else:
                lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
                lemma_rew = lemma_rew + " " + lemma
        return lemma_rew


def preprocess_tweets(df)->pd.DataFrame:
    '''
    Function takes in the whole dataframe and carries out the following preprocessing steps:

    1. General text cleaning
    2. Part of Speech tagging
    3. Lemmatization

    Then return the dataframe
    ''' 
    #max_date = data.Datetime.max() 
    #if max_date is not None:
    #    print(f"Fetching extra records as from {max_date}")
    #    fetched_data = fetch_sectioned_tweets(max_date)
        #df=fetched_data
    #    df = data.append(fetched_data)
    #    print("Done fetching  extra records!!")
    #else:
    #    print("Fetching all records")
    #    df = fetch_all_tweets()
    #    print("Done fetching records...") 

    df['clean_text'] = df['message'].apply(lambda x:_clean_tweet(x))
    df['POS tagged'] = df['clean_text'].apply(_token_stop_pos)
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    df['hashtags'] = df['message'].apply(lambda x: " ".join ([w for w in x.split() if '#'  in w[0:3] ]))
    df['hashtags']=df['hashtags'].str.replace("[^a-zA-Z0–9]", ' ')
    df = df.loc[:,['clean_text', 'POS tagged','Lemma','hashtags', "sentiment"]]
    return df



def preprocess_test(df)->pd.DataFrame:
    '''
    Function takes in the whole dataframe and carries out the following preprocessing steps:

    1. General text cleaning
    2. Part of Speech tagging
    3. Lemmatization

    Then return the dataframe
    ''' 
    #max_date = data.Datetime.max() 
    #if max_date is not None:
    #    print(f"Fetching extra records as from {max_date}")
    #    fetched_data = fetch_sectioned_tweets(max_date)
        #df=fetched_data
    #    df = data.append(fetched_data)
    #    print("Done fetching  extra records!!")
    #else:
    #    print("Fetching all records")
    #    df = fetch_all_tweets()
    #    print("Done fetching records...") 

    df['clean_text'] = df['message'].apply(lambda x:_clean_tweet(x))
    df['POS tagged'] = df['clean_text'].apply(_token_stop_pos)
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    df['hashtags'] = df['message'].apply(lambda x: " ".join ([w for w in x.split() if '#'  in w[0:3] ]))
    df['hashtags']=df['hashtags'].str.replace("[^a-zA-Z0–9]", ' ')
    df = df.loc[:,['clean_text', 'POS tagged','Lemma','hashtags']]
    return df