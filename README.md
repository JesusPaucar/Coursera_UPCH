# Coursera_UPCH
Mapeo automático de semejanzas de cursos basado en el modelo NLP **paraphrase-multilingual-MiniLM-L12-v2**

## Acerca del modelo

El modelo __paraphrase-multilingual-MiniLM-L12-v2__ es una herramienta de procesamiento de lenguaje natural que está diseñada para la generación de sinónimos y el parafraseo de textos en varios idiomas, es capaz de interpretar, parafrasear textos completos para relacionarlos y encontrar la similaridad entre textos. Puede interpretar alrededor de 50 idiomas distintos incluyendo español, inglés, francés, alemán, etc.

Para más información: [Link modelo NLP](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

## Sistema operativo

1. Ubuntu 22.04.2 LTS (GNU/Linux 5.15,0-76-generic x86_64)
   1.1. 4 CPU(s)
   1.2. 8 GB RAM
2. Anaconda3: [Link to download](https://www.anaconda.com/download)

## Dependencias

pandas == 1.5.3
numpy == 1.23.5
pdfplumber == 0.8.0
cleantext[gpl] == 0.6.0
streamlit == 1.20.0
re == 2.2.1
nltk == 3.7
sentence_transformers == 2.2.2
torch == 2.0.0+cpu
skimage == 0.19.3
keybert == 0.7.0

## Comandos de instalación

wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
pip install transformers
pip install sentence-transformers
pip install pdfplumber
pip install clean-text[gpl]
pip install keybert
pip install streamlit
