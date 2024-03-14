from data_loader.data_loading import load_credits, load_links, load_keywords

def merge_links(movies):
    links = load_links()
    movies = movies.merge(links, left_on='id', right_on='tmdbId', how='left')
    movies.drop('tmdbId', axis=1, inplace=True)
    return movies

def merge_credits(movies):
    credits = load_credits()
    credits.columns = ['cast', 'crew', 'id']
    movies = movies.merge(credits, on='id')
    return movies

def merge_keywords(movies):
    keywords = load_keywords()
    movies = movies.merge(keywords, on='id')
    return movies