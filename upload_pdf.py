import pandas as pd
import numpy as np
import pdfplumber
from cleantext import clean
import streamlit as st
import re
import nltk
nltk.download('punkt')
from sentence_transformers import SentenceTransformer, util
import torch
from skimage import io
import keybert

def get_text(pdf_filename):
  text = ''
  with pdfplumber.open(pdf_filename) as pdf:
    for p in pdf.pages:
      text += p.extract_text()
  text = clean(text)
  
  return text

def data_preprocessing(review):
  
  review = re.sub(re.compile('<.*?>'), '', review)
  review = re.sub('[^A-Za-z0-9]+', ' ', review)
  
  review = review.lower()
  
  tokens = nltk.word_tokenize(review)
  
  review = ' '.join(tokens)
  return review

@st.cache_resource
def get_model(name = 'paraphrase-multilingual-MiniLM-L12-v2'):
    return SentenceTransformer(name)

@st.cache_resource
def get_keybert(name = 'paraphrase-multilingual-MiniLM-L12-v2'):
    return keybert.KeyBERT(model = name)

def prediction(target):
  model = get_model()
  embedding1 = model.encode(finish_content, convert_to_tensor=True)
  embedding2 = torch.load('reference2.pt')
  cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
  max_sim = torch.topk(cosine_scores, 10)
  a = max_sim[0].detach().cpu().numpy()
  
  courses_neigh = ['Objetivo']
  description = [finish_content]
  a1, b1, c1, d1, e1, f1 = ['NA'], ['NA'], ['NA'], ['NA'], ['NA'], ['UPCH']
  for idx in range(max_sim[1][0].shape[0]):
      courses_neigh.append(courses[max_sim[1][0][idx].item()])
      description.append(original_text[max_sim[1][0][idx].item()])
      a1.append(level[max_sim[1][0][idx].item()])
      b1.append(average[max_sim[1][0][idx].item()])
      c1.append(rating[max_sim[1][0][idx].item()])
      d1.append(URL[max_sim[1][0][idx].item()])
      e1.append(hrange[max_sim[1][0][idx].item()])
      f1.append(university[max_sim[1][0][idx].item()])

  dictionary_result = {'%Similaridad': np.append(0, a[0])*100, 'Curso': courses_neigh, 'Universidad': f1, 'Dificultad': a1, 'Hrs_Promedio': b1, 'Ranking': c1, 'Link': d1, 'Descripcion': description, 'Hrs_Sem': e1}
  df_result = pd.DataFrame(dictionary_result)

  return df_result

@st.cache_data
def lectura(filename = 'UPCH C4C Best Practice Curations Healthcare and Life Sciences.xlsx', hoja = 1):
    df = pd.read_excel(filename, hoja, skiprows = 4).iloc[:, 1:]
    df.drop_duplicates(subset = 'Course Name', keep = False, inplace = True, ignore_index = True)
    return df



logo = io.imread('UPCH.png')
st.image(logo, width = None, use_column_width = True, clamp = False, channels = 'RGB', output_format = 'png')
st.markdown("<h1 style='text-align: center; color: grey;'>Aplicación de búsqueda de cursos - Coursera</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: left; color: grey;'>Descripción del modelo de búsqueda: </h2>", unsafe_allow_html=True)

st.markdown("<h6 style = 'text-align': justify; color : black; '>El modelo 'paraphrase-multilingual-MiniLM-L12-v2' es una herramienta de procesamiento de lenguaje natural que está diseñada para la generación de sinónimos y la paraphrase de textos en varios idiomas, es capaz de parafrasear textos completos para relacionarlos y encontrar la similaridad entre textos. Puede interpretar alrededor de 50 idiomas distintos incluyendo español, inglés, francés, alemán, etc.</h6>",
            unsafe_allow_html = True)

st.markdown("<h6 style = 'text-align': justify; color : black; '>Por otro lado, tendremos como referencia de búsqueda tenemos los cursos relacionados a las ciencias de la salud y vida ofrecidos por la plataforma de COURSERA</h6>",
            unsafe_allow_html = True)

st.markdown("<h2 style='text-align: left; color: grey;'>Búsqueda: </h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: justify; color: black;'>Se puede realizar una búsqueda de dos formas distintas</h3>", unsafe_allow_html=True)


st.markdown("<h4 style='text-align: center; color: black;'>1. Para realizar la búsqueda es necesario subir el documento PDF del sílabo</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader('Choose yout .pdf file', type = 'pdf', disabled = False)

st.markdown("<h4 style='text-align: center; color: black;'>2. Búsqueda por entrada específica, puede introducir el texto que se desee</h4>", unsafe_allow_html=True)

text = st.text_area('Introducir texto específico a buscar', disabled = False)

df = lectura()
    
documents = []
original_text = []
courses = []
university = []
level = []
average = []
rating = []
URL = []
hrange = []
lineas_error = 0 
for index,t in enumerate(df['Course Description']):
  try:
    documents.append(data_preprocessing(t))
    original_text.append(t)
    courses.append(df['Course Name'].values[index])
    university.append(df['University / Industry Partner Name'].values[index])
    level.append(df['Difficulty Level'].values[index])
    average.append(df['Average Hours'].values[index])
    rating.append(df['Course Rating'].values[index])
    URL.append(df['Course URL'].values[index])
    hrange.append(df['Hours Range'].values[index])
  except:
    lineas_error += 1
   

if uploaded_file is not None:
  silabo = get_text(uploaded_file)
  content = data_preprocessing(silabo)
  idx_u1 = content.find('unidad 1')
  idx_finish = content.find('v estrategias')
  finish_content = content[idx_u1: idx_finish]
  df_result = prediction(target = finish_content)
  st.markdown("<h6 style = 'text-align': justify; color : black; '>Los cursos más cercanos al curso objetivo (consultado) son: </h6>",
                unsafe_allow_html = True)
  st.dataframe(df_result)
  
  st.markdown("<h6 style = 'text-align': justify; color : black; '>Los términos importantes del curso similar son: </h6>",
                unsafe_allow_html = True)
  
  button = st.checkbox('Ver Keywords')
  
  if button:
      kw_model = get_keybert()
      options = df_result['Curso'].values
      descrip = df_result['Descripcion'].values
      for i in range(1, options.shape[0]):
          st.write('Curso ' + str(i) + ': ' + options[i])
          ddd = pd.DataFrame(kw_model.extract_keywords(descrip[i], keyphrase_ngram_range = (1, 1), stop_words = None, highlight = False))
          st.dataframe(ddd)
  
  
elif text is not None:
  finish_content = text.splitlines()
  finish_content = clean(finish_content)
  df_result = prediction(target = finish_content)
  st.markdown("<h6 style = 'text-align': justify; color : black; '>Los cursos más cercanos al curso objetivo (consultado) son: </h6>",
                unsafe_allow_html = True)
  st.dataframe(df_result)
  
  st.markdown("<h6 style = 'text-align': justify; color : black; '>Los términos importantes del curso similar son: </h6>",
                unsafe_allow_html = True)
  
  button2 = st.checkbox('Ver Keywords')
  
  if button2:
      kw_model2 = get_keybert()
      options2 = df_result['Curso'].values
      descrip2 = df_result['Descripcion'].values
      for i in range(1, options2.shape[0]):
          st.write('Curso ' + str(i) + ': ' + options2[i])
          ddd2 = pd.DataFrame(kw_model2.extract_keywords(descrip2[i], keyphrase_ngram_range = (1, 1), stop_words = None, highlight = False))
          st.dataframe(ddd2)
