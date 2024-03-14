import pandas as pd
import os


def load_credits():
    credits = pd.read_csv('data/credits.csv')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    return credits

def load_movies():
    movies = pd.read_csv('data/movies_metadata.csv', dtype={'genres': 'object'}, low_memory=False)
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    return movies

def load_links():
    # Verificar la existencia del archivo CSV
    file_path = 'data/links.csv'
    if not os.path.exists(file_path):
        return None
    else:
        links = pd.read_csv('data/links.csv')
        links.drop('movieId', axis=1, inplace=True)
        return links

def load_keywords():
    # Verificar la existencia del archivo CSV
    file_path = 'data/keywords.csv'
    if not os.path.exists(file_path):
        return None
    else:
        keywords = pd.read_csv(file_path)
        return keywords