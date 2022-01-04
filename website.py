from os import write
import streamlit as st
from underthesea import word_tokenize
import pandas as pd
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import time
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from PIL import Image

def preprocess_text(text):
     text = text.strip()
     with open('stopwords.txt','r',encoding='utf-8') as stopfin:
          stopwords = stopfin.readlines()
          for i in range(len(stopwords)):
               stopwords[i] = word_tokenize(stopwords[i].strip(), format = 'text')
          
     patweb = '[http\:\/\/|https\:\/\/]+[www\.]*\w+\.[\w+.|\w+\/\w+\-?%&=]+'

     special_char = ['@','#', '/','!','.',',','\\','\'','"','+','-','=',':',';','...','(',')','“','”']

     after_processed = []
     text = word_tokenize(text,format = 'text').split(' ')
     print(str(i) + ' | ' + str(len(text)),end=' | ')
     for word in text:
          word = word.lower()
          url = re.findall(patweb,word)
          if  word not in special_char:
               if word not in url:
                    if word not in stopwords:
                         after_processed.append(word)
     text_after = ' '.join(after_processed)

     return text_after


image = Image.open("Image.jpg")
st.image(image, caption="Tech Tent: Social media fights a fresh flood of fake news - BBC News")

st.write("""
# Fake news Detection Website
""")

st.subheader("""
Input News to analyze
""")
news = st.text_area(label="News to detect Fake or Real")


st.subheader("""
Choose a Model for detecting
""")


st.sidebar.write("""
# About us

**You can visit here for more information [link]()**

## What is this
> This is our class-project use **sklearn** package to build a model to detect which news is fake and which news is real.
""")


option = st.selectbox(
     '',
     ['Multinomial NB', 'SGD Classifier', 'Random Forest Classifier']
)

st.markdown("""
#### You selected: {}
""".format(option))

st.subheader("""
Detecting
""")

click = st.button("""
Detect
""")

if click:
     if len(news) != 0:
          model = None
          if option == 'Multinomial NB':
               model = joblib.load("mnbc.mdl")
          elif option == 'SGD Classifier':
               model = joblib.load('sgdc.mdl')
          elif option == 'Random Forest Classifier':
               model = joblib.load('rdfc.mdl')

          label = model.predict([preprocess_text(news)])

          if label == 0:
               st.write("This is fake news!!!")
          else:
               st.write("This is real news!!!")

     else:
          st.warning("You haven't entered text!!!")
else:
     pass
# click = False