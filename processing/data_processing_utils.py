import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in words if word.lower() not in stop_words and word.isalpha()])

def process_genres(movies):
    movies['genres'] = movies['genres'].apply(fix_json_format)
    movies['genres'] = movies['genres'].apply(lambda x: ",".join([genre['name'] for genre in x]))   
    movies['genres'] = movies['genres'].fillna('').apply(preprocess_text)   
    return movies

def process_overview(movies):
    movies['overview'] = movies['overview'].fillna('').apply(preprocess_text)
    movies.drop('crew', axis=1, inplace=True)
    return movies

def process_cast(movies):
    movies['cast'] = movies['cast'].apply(fix_json_format)
    movies['cast'] = movies['cast'].apply(lambda x: ",".join([actor['name'] for actor in x]))
    movies['cast'] = movies['cast'].fillna('').apply(preprocess_text)   
    return movies

def fix_json_format(json_str):
    try:
        return json.loads(json_str.replace("'", "\""))
    except json.JSONDecodeError:
        return []
    
def process_keywords(movies):
    movies['keywords'] = movies['keywords'].apply(fix_json_format)
    movies['keywords'] = movies['keywords'].apply(lambda x: ",".join([keyword['name'] for keyword in x]))   
    movies['keywords'] = movies['keywords'].fillna('').apply(preprocess_text)   
    return movies

def combine_text_columns(movies):
    movies['combined_text'] = movies['overview'] + ' ' + movies['cast'] + ' ' + movies['genres']+ ' ' + movies['keywords']
    return movies

def calculate_score(movies):
    C = movies['vote_average'].mean()
    percentil = movies['vote_count'].quantile(0.8)
    movies['score'] = movies.apply(rating_ponderado, axis=1, args=(percentil, C))
    return movies

def rating_ponderado(x, m, c):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * c)