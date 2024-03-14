import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(movies):
    # Exploración inicial de los datos
    median_average = movies['vote_average'].median()
    median_count = movies['vote_count'].median()
    percentil = movies['vote_count'].quantile(0.8)

   
    # Aplicar el filtro y obtener el nuevo DataFrame
    d_movies = movies.copy().loc[movies['vote_count'] > percentil]
     # Imprimir información adicional
    print(f"Número total de películas: {len(movies)}")
    registros, columnas = d_movies.shape
    # Imprimir información sobre el nuevo DataFrame
    print(f"Número de películas después del filtro: {len(d_movies)}")
    print("Estadísticas de 'vote_count':")
    print(d_movies['vote_count'].describe())
    print("Estadísticas de 'vote_average':")
    print(d_movies['vote_average'].describe())

    # Visualización con Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='vote_count', y='vote_average', data=d_movies)
    plt.title("Distribucion de Promedio de Votos")
    plt.xlabel("Conteo")
    plt.ylabel('Promedio de Votos')
    plt.show()

    popular_movies = d_movies.sort_values('popularity', ascending=False)

    plt.figure(figsize=(12,4))

    plt.barh(popular_movies['title'].head(3), popular_movies['popularity'].head(3), align='center', color='red')
    plt.xlabel('Popularidad')
    plt.ylabel('Película')
    plt.title('Películas más populares')
    plt.show()
