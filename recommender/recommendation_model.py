import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rating_ponderado(data, percentil, C):
    # Lógica para calcular el rating ponderado
    v = data['vote_count']
    R = data['vote_average']
    return (v / (v + percentil) * R) + (percentil / (percentil + v) * C)

def train_recommender(movies):
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

def entreno(movies):
    try:
        tfidf_matrix = load_tfidf_matrix()
    except:
        tfidf_matrix = train_recommender(movies)
    return tfidf_matrix

def recommend(reference_movie_index):
    print(f"def recommend")
    print(reference_movie_index)
    try:
        tfidf_matrix = load_tfidf_matrix()
    except:
        tfidf_matrix = train_recommender(movies)
    return tfidf_matrix
    # Resto del código de recomendación
    cosine_similarity = cosine_similarity(tfidf_matrix[reference_movie_index], tfidf_matrix)
    print(f"cosine_sim_scores: {cosine_similarity}")

    # Obtener los índices de las películas con mayor similitud (excluyendo la película de referencia)
    peliculas_similares = cosine_similarity.argsort()[0][::-1][1:]
    print(f"Similar movie indices: {peliculas_similares}")
    # Obtener las principales N películas similares
    top_N = 5
    recommended_movies = movies.iloc[peliculas_similares[:top_N]]
    print(f"recommended_movies.columns: {recommended_movies.columns}")
    return "\n".join(recommended_movies['title'].tolist())
