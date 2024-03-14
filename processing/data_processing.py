from data_cleaning.data_cleaning import fill_column_with_chatgpt
from processing.data_processing_utils import process_genres, process_cast, process_keywords, process_overview, combine_text_columns, calculate_score
from processing.data_merging import merge_links, merge_credits, merge_keywords


def preprocess_data(movies):
    movies = process_genres(movies)
    movies = merge_links(movies)
    movies = merge_credits(movies)
    movies = merge_keywords(movies)
    movies = process_overview(movies)
    movies = process_cast(movies)
    movies = process_genres(movies)
    movies = process_keywords(movies)
    movies = combine_text_columns(movies)
    movies = calculate_score(movies)
    #print(movies.info())
    return movies

def fill_missing_dates(movies):
    for index, row in movies.iterrows():
        filled_value = fill_column_with_chatgpt(row, 'release_date')
        movies.at[index, 'release_date'] = filled_value

    return movies

def remove_unwanted_columns(movies):
    columns_to_drop = ['adult', 'homepage', 'imdb_id', 'tagline', 'poster_path', 'belongs_to_collection', 'budget', 'genres', 
                       'homepage', 'id', 'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 
                       'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status',
                       'tagline', 'video', 'imdbId', 'cast', 'score', 'keywords']
    movies = movies.drop(columns_to_drop, axis=1)
    return movies