from processing.data_processing import preprocess_data, remove_unwanted_columns
from data_exploration.explore_data import explore_data
from recommender.recommendation_model import recommend,load_tfidf_matrix,train_recommender
import gradio as gr
from data_loader.data_loading import load_movies
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_utils.excel_utils import save_to_excel
import streamlit as st

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)
# Cargar películas
movies = load_movies()

# Procesar datos
movies = preprocess_data(movies)
movies = remove_unwanted_columns(movies)

save_to_excel(movies)

# Explorar datos
explore_data(movies)

def rating_ponderado(data, percentil, C):
    # Lógica para calcular el rating ponderado
    v = data['vote_count']
    R = data['vote_average']
    return (v / (v + percentil) * R) + (percentil / (percentil + v) * C)

def train_recommender():
    # Lógica para entrenar el modelo de recomendación
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_text'])
    save_recomendation(tfidf_matrix)
    return tfidf_matrix

def save_recomendation(tfidf_matrix):
    with open('tdidf_matrix.pickle', 'wb') as handle:
        pickle.dump(tfidf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tfidf_matrix():
    with open('tdidf_matrix.pickle', 'rb') as handle:
        tfidf_matrix = pickle.load(handle)
    return tfidf_matrix

def train():
    try:
        tfidf_matrix = load_tfidf_matrix()
    except Exception as e:
        print(f"Error al cargar tfidf_matrix: {e}")
        tfidf_matrix = train_recommender()
    return tfidf_matrix

def recommend(reference_movie_index):
    print(f"def recommend")
    print(reference_movie_index)
    tfidf_matrix = train()
    try:
        # Resto del código de recomendación
        cosine_similarities  = cosine_similarity(tfidf_matrix[reference_movie_index], tfidf_matrix)
        print(f"cosine_sim_scores: {cosine_similarity}")
        # Obtener los índices de las películas con mayor similitud (excluyendo la película de referencia)
        peliculas_similares = cosine_similarities.argsort()[0][::-1][1:]
        print(f"Similar movie indices: {peliculas_similares}")
        # Obtener las principales N películas similares
        top_N = 5
        recommended_movies = movies.iloc[peliculas_similares[:top_N]]
        print(f"recommended_movies.columns: {recommended_movies.columns}")
        return "\n".join(recommended_movies['title'].tolist())
    except Exception as e:
            print(f"Error en el cálculo de la recomendación: {e}")


options = [(row["title"], index ) for index, row in movies.iterrows()][:50]

print(options[46891:46897])

demo = gr.Interface(
    recommend,
    [
        gr.Dropdown(
            choices=options, label="Pelicula", info="Selecciona una pelicula que te haya gustado"
        ),

    ], 
    "text"
)


demo.launch(debug=True)
