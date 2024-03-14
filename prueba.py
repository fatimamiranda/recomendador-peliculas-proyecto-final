#!pip install gradio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json 
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import pickle


def explore_data(movies):

    #Exploracion inicial de los datos
    median_avarage = movies['vote_average'].median()
    median_count = movies['vote_count'].median()
    #print(median_average, median_count)
    percentil = movies['vote_count'].quantile(0,8)
    #print(percentil)
    # print("Numero de peliculas dentro del percentil: " +str(len(movies['vote_count']>percentil[1])))


    d_movies = movies.copy().loc[movies['vote_count'] > percentil]
    registros, columnas = d_movies.shape

    #print("registros: ")

    #Con Seaborn
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='vote_count', y='vote_average', data=d_movies)
    plt.title("Distribucion de Promedio de Votos")
    plt.xlabel("Conteo")
    plt.ylabel('Promedio de Votos')
    plt.show()

    #Podemos hacer una visualizacion de las peliculaes con mayor indice de popularidad
    populares = d_movies.sort_values('popularity', ascending=false)

    plt.figure(figsize=(12,4))

    plt.barh(populares['title'].head(6).populares['popularity'].head(6), align='center', color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Popularidad")
    plt.title("Peliculas Populares")
    plt.show()

# El indice bayes, que se utiliza en el sistem Imbd nos permite equilibrar las valoraciones con el numero de votos
# v es el numero de votos por pelicula (vote_count)   .
# m es el umbral minimo de votos de nuestro caso el percentil 0.8.
# R es la calificacion promedio de la pelicula (vote_average)
# C es el promedio de votos general. d_movies['vote_average'].mean()
def rating_ponderado(data, percentil, C):

    v = data['vote_count']
    R = data['vote_average']
    return ( v/(v+percentil) * R) +(percentil/(percentil+v) * C)

# Mostramos las primeras 10 con la columnas
# print(movies[['title'], 'score' , 'vote_count' , 'vote_average' ]].head(10))

def preprocess(text):

    #Preparamos stemer y stopwords
    stemer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    #Tokenizar el texto
    words = nltk.word_tokenize(text)

    # Eliminar stopwords y aplicar steming 
    return ' '.join([stemer.stem(word) for word in words if word.lower() not in stop_words and word.isalpha()])


def load_data():
    # Carga del dataset de creditos
    credits = pd.read_csv('data/credits.csv')

    # Carga del dataset de peliculas
    movies = pd.read_csv('data/movies_metadata.csv')

    #Extraer el "name" de la lista de diccionarios
    movies['genres'] = movies['genres'].apply(lambda a: [genre['name'] for genre in x])
    movies['genres'] = movies['genres'].apply(lambda x: ",".join(x))

    #Cargar el dataset de equivalencias 
    links = pd.read_csv('data/links.csv')
    links.drop('movieId', axis = 1, inplace=True)

    movies = movies.merge(links, left_on = 'id', right_on='tmbId', how='left')
    movies.drop('tmbId', axis=1, inplace=True)

    credits.columns = ['id', 'title', 'cast', 'crew']
    credits.drop('title', axis = 1, inplace=True)
    movies = movies.merge(credits, on='id')

    #con fillna ponemos strings vacias en NaN , aply 
    #pasamos la funcion sobre cada overview
    movies['overview'] = movies['overview'].fillna('').apply(preprocess)

    #Descartamos crew porque n aporta y quitamos  peso
    movies.drop('crew', axis = 1, inplace=True)

    #convert the string representation to a list of dictionaries
    movies['cast'] = movies['cast'].apply(lambda x:json.loads(x)) 

    #Extract the name values from the list of dictionaries
    movies['cast'] = movies['cast'].apply(lambda x: [actor['name'] for actor in x])
    movies['cast'] = movies['cast'].apply(lambda x: ",".json(x))

    #Convertir el json a un list de diccionarios
    movies['genres'] = movies['genres'].apply(lambda x: json.loads(x))

    #Extract the name values from the list of dictionaries
    movies['genres'] = movies['genres'].apply(lambda x: [genre['name'] for genre in x])
    movies['genres'] = movies['genres'].apply(lambda x: ",".json(x))

    #Convertir el json a un list de diccionarios
    movies['keywords'] = movies['keywords'].apply(lambda x: json.loads(x))

    #Extract the 'name' values from the list of dictionaries
    movies['keywords'] = movies['keywords'].apply(lambda x: [keyword['name'] for keyword in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: ",".json(x))

    #Combine the text from "cast," "genres," and "keywords" columns into a single text column
    movies['combined_text'] = movies['overview'] + '' + movies['cast'] + ' ' + movies['genres'] + ' ' + movies['kewywords']

    #Promedios de votos lo vamos a necesitar para calcular indice bayes
    c = movies['vote_avarage'].mean()

    #Agregamos a movies el campo score que nos servira para ordenar los resultados
    # y ofrecer los mejor valores primero, d_movies es nuestro dataframe con percentil

    percentil = movies['vote_count'].quantile(0,8)
    movies['score'] = movies.apply(rating_ponderado, axis=1, args=(percentil, C))

    #Ordenar pelicula en descente por score
    return movies

def save_to_excel(movies):
    #Guardamos a excel
    movies.to_excel('excel_movies.xlsx', index=True)

movies = load_data()
save_to_excel(movies)

def train_recomendation():

    # Creamos un TF-IDF vectorizer
    tdidf_vectorizer = TfidfVectorizer()

    #Fit y transform la columna combinada
    tdidf_matrix = tdidf_vectorizer.fit_transform(movies['combined_text'])

    save_recomendation(tdidf_matrix)
    return tdidf_matrix

def save_recomendation(tfidf_matrix):
    with open('tfidf_matrix.pickle', 'wb') as handle:
        pickle.dump(tfidf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tfidf_matrix():

    with open('tdidf_matrix.pickle', 'rb') as handle:
        tdfidf_matrix = pickle.load(handle)
    return tdfidf_matrix

def recomendation(reference_movie_index):
    #Si existe el fichero con la matriz lo cargamos para no volver a entrenar, si no llamamos a train_recommendation
    try:
        tdidf_matrix = load_tfidf_matrix()
    except:
        tfidf_matrix = train_recomendation()

    cosine_sim_scores = cosine_similarity(tdidf_matrix[reference_movie_index], tfidf_matrix)

    #Get the indices of movies with highest similarity (excluding the reference movie)
    similar_movie_indices = cosine_sim_scores.argsort()[0][::-1][1:]

    #Get the top N similar movies
    top_N = 10
    recomended_movies = movies.iloc[similar_movie_indices[:top_N]]
    print(recomended_movies.columns)

#Crea una lista de diccionarios para el dropdown
movie_options = [(row["title"], index) for index, row in movies.iterrows()]

demo = gr.Interface(
    recomendation,
    [
        gr.Dropdown(
            movie_options, 
            label = "Pelicula", 
            info="Selecciona una pelicula que te haya gustado"
        )
    ], "text"
)

demo.launch()