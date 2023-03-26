import pandas as pd
import numpy as np
import pdfplumber
import translators as ts
import translators.server as tss
from cleantext import clean
import streamlit as st
#import re
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
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

uploaded_file = st.file_uploader('Choose yout .pdf file', type = 'pdf')
if uploaded_file is not None:
  df = extract_data(uploaded_file)
  silabo = get_text(df, translate = True)

st.write(silabo)
