import pandas as pd
import numpy as np
import pdfplumber
import translators as ts
import translators.server as tss
from cleantext import clean
import streamlit as st
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from google.colab import files
#from sentence_transformers import SentenceTransformer, util

def get_text(pdf_filename, translate = False):
  text = ''
  with pdfplumber.open(pdf_filename) as pdf:
    for p in pdf.pages:
      text += p.extract_text()
  text = clean(text)
  if translate:
    n = 2000
    text = [text[i:i+n] for i in range(0, len(text), n)]
    text = [tss.google(t, 'es', 'en',) for t in text]
    text = ' '.join(text)
  return text

def data_preprocessing(review):
  stop_words = stopwords.words('english')
  lemmatizer = WordNetLemmatizer()
  
  review = re.sub(re.compile('<.*?>'), '', review)
  review = re.sub('[^A-Za-z0+9]+', ' ', review)
  
  review = review.lower()
  
  tokens = nltk.word_tokenize(review)
  review = [word for word in tokens if word not in stop_words]
  review = [lemmatizer.lemmatize(word) for word in review]
  review = ' '.join(review)
  return review

uploaded_file = st.file_uploader('Choose yout .pdf file', type = 'pdf')
if uploaded_file is not None:
  #df = extract_data(uploaded_file)
  silabo = get_text(uploaded_file, translate = True)
  content = data_preprocessing(silabo)
  idx_u1 = content.find('unit 1')
  #idx_u4 = content.find('unit 4')
  idx_finish = content.find('v didactic')
  #print(content[idx_u1: idx_u1 + 10])
  #print(content[idx_u4: idx_u4 + 10])
  #print(idx_finish)
  #print(content[idx_finish: idx_finish + 5])
  finish_content = content[idx_u1: idx_finish]
  st.write(finish_content)
