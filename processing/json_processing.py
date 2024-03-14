import json

def fix_json_format(json_str):
    try:
        return json.loads(json_str.replace("'", "\""))
    except json.JSONDecodeError:
        return []

def process_json_columns(movies, column_name):
    movies[column_name] = movies[column_name].apply(fix_json_format)
    movies[column_name] = movies[column_name].apply(lambda x: ",".join([item['name'] for item in x]))
    return movies


