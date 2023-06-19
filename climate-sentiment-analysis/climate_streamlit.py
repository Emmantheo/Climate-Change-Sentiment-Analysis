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
import requests
from PIL import Image

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


def main():
    # Get the image from a URL
    image1_url = "https://i.guim.co.uk/img/media/377e5731a1651bcdeea9d3d519b8e7786c2c7fdb/0_133_4000_2400/master/4000.jpg?width=620&quality=45&dpr=2&s=none"
    response = requests.get(image1_url, stream=True)
    image = Image.open(response.raw)

    # Reduce the height and width of the image
    new_width1 = 400  # Desired width
    new_height1 = 100  # Desired height
    resized_image = image.resize((new_width1, new_height1))

    # Display the resized image in Streamlit
    st.image(resized_image, use_column_width=True)


    st.markdown('##### What is the public perception of climate change?')
    
    message1 = st.text_area("Type in or paste any text segment (e.g. publication excerpt, news article) in the text box below", "Type Here")
    if st.button("Get Sentiment"):
        with st.spinner('Running model...'):
            time.sleep(1)
        vect_text = vectorizer.transform([message1]).toarray()
        predicted = classifier.predict(vect_text)
        #clean_message = process_message(message1)
        #result1 = text_predict(clean_message)
        word = ''
        if predicted == 0:
            word = '"# Neutral". It neither supports nor refutes the belief of man-made climate change'
        elif predicted == 1:
            word = '"# Pro". The tweet supports the belief of man-made climate change'
        elif predicted == 2:
            word = '**News**. The tweet links to factual news about climate change'
        else:
            word = 'The tweet do not belief in man-made climate change'

        st.success("Text Categorized as: {}".format(word))
 

     


if __name__ == '__main__':
    main()


