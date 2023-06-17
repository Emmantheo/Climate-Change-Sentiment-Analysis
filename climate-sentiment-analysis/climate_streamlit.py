import streamlit as st
import plotly.express as px
import numpy as np 
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import texthero as hero
from kedro.config import ConfigLoader
from kedro.framework.project import settings
import logging.config
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

from kedro.io import DataCatalog
import yaml

from kedro.extras.datasets.pickle import PickleDataSet

from kedro.extras.datasets.pandas import (
    CSVDataSet,
    SQLTableDataSet,
    SQLQueryDataSet,
    ParquetDataSet,
)

config = {
    "cleaned_train_data": {
        "type": "pandas.CSVDataSet",
        "filepath": "data/02_intermediate/cleaned_train_data.csv",
        "load_args": {
            "sep": ','
        }
    },
    "tweet_classifier": {
        "type": "pickle.PickleDataSet",
        "filepath": "data/06_models/tweet_classifier.pickle",
        "backend": "pickle"
    },
    "vectorizer": {
        "type": "pickle.PickleDataSet",
        "filepath": "data/06_models/vectorizer.pickle",
        "backend": "pickle"
    }, 
    "predictions": {
        "type": "pandas.CSVDataSet",
        "filepath": "data/07_model_output/predictions.csv",
        "load_args": {
            "sep": ','
        }
    },
}


#retieving keys and secret
conf_path = "conf/"
conf_loader = ConfigLoader(conf_path)
conf_catalog = conf_loader.get("data*")

catalog = DataCatalog.from_config(config, conf_catalog)


#cache function that loads in data
@st.cache(allow_output_mutation = True)
def load_data(data_name):
    data = catalog.load(data_name)
    #can add extra stuff here
    return data


data_load_state = st.text('Loading data from data directory...')
data = load_data("train_data")
#catalog.save("boats", df)

data_load_state.text("")

#load in classifier and vectorizer models
classifier = catalog.load("tweet_classifier")
vectorizer = catalog.load("vectorizer")

def process_message(message):

    '''
    TO DO:
    function that cleans input text data
    '''

    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the message
    doc = nlp(message)

    # Example NLP processing step
    transformed_message = " ".join([token.lemma_ for token in doc])  # Lemmatization

    # Return the transformed message
    return transformed_message

def text_predict(my_text):
    '''
    Takes in text (from text box - string), then vectorizes the text and then applies
    the trained linSVC model to get SDG label predictions
    '''
    #doc = list(my_text.split(" "))
    doc = vectorizer.transform(my_text)
    predicted = classifier.predict(doc)

    return predicted

def main():
    
    st.markdown('##### What is the public perception of climate change?')
    
    message1 = st.text_area("Type in or paste any text segment (e.g. publication excerpt, news article) in the text box below", "Type Here")
    if st.button("Get Sentiment"):
        with st.spinner('Running model...'):
            time.sleep(1)
        clean_message = process_message(message1)
        result1 = text_predict(clean_message)
 

     
    

if __name__ == '__main__':
    main()


