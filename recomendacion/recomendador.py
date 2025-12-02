# recomendador.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dp_alineamiento import needleman_wunsch_score
from ordenamiento import quicksort_by_score
from grafos import UsuarioPeliculaGrafo
from posters import PosterFetcher

class Recomendador:
    def __init__(self, data_folder="data"):
        # Cargar archivos CSV
        self.movies = pd.read_csv(f"{data_folder}/movies.csv")
        self.ratings = pd.read_csv(f"{data_folder}/ratings.csv")

        # 1. Contamos cuantos ratings tiene cada pelicula
        conteo_ratings = self.ratings['movieId'].value_counts()
        
        # 2. limitamos solo a 2500 peliculas
        top_movies_ids = conteo_ratings.head(2500).index
        
        # 3. Filtramos los dataframes para borrar el resto
        self.movies = self.movies[self.movies['movieId'].isin(top_movies_ids)]
        self.ratings = self.ratings[self.ratings['movieId'].isin(top_movies_ids)]

        # Procesa los generos one-hot (ahora con menos datos)
        self.genres_dummies = self.movies["genres"].str.get_dummies(sep="|")
        self.movies = pd.concat([self.movies, self.genres_dummies], axis=1)

        # Promedio rating por movie
        movie_ratings = self.ratings.groupby("movieId")["rating"].mean().rename("avg_rating")
        self.movies = self.movies.merge(movie_ratings, on="movieId", how="left")
        
        # Advertencia de Pandas
        self.movies["avg_rating"] = self.movies["avg_rating"].fillna(0)

        # Crea el grafo usuario-pelicula
        self.grafo = UsuarioPeliculaGrafo(self.ratings)

        # Muestra los Poster
        self.poster_fetcher = PosterFetcher()

    def get_all_genres(self):
        return list(self.genres_dummies.columns)

    def get_user_ratings(self, user_id):
        # Muestra las peliculas que califico el usuario y su rating
        ur = self.ratings[self.ratings["userId"] == user_id].merge(
            self.movies[["movieId","title","avg_rating"]], on="movieId", how="left"
        )
        return ur.sort_values(by="rating", ascending=False).to_dict(orient="records")

    def recomendar_peliculas(self, user_id, n=3, filtro_genero=None, min_rating=0, year=None):
        # Ratings del usuario
        user_data = self.ratings[self.ratings["userId"] == user_id]
        if user_data.empty:
            return []

        peliculas_vistas = set(user_data["movieId"].tolist())

        # Vector de preferencias por genero
        watched_movies = self.movies[self.movies["movieId"].isin(peliculas_vistas)]
        if watched_movies.empty:
            user_genre_vector = np.zeros(len(self.genres_dummies.columns)).reshape(1,-1)
        else:
            # Ponderar por rating
            merged = watched_movies.merge(user_data[["movieId","rating"]], on="movieId")
            weights = merged["rating"].values.reshape(-1,1)
            genre_matrix = merged[self.genres_dummies.columns].values
            weighted = (genre_matrix * weights).sum(axis=0)
            user_genre_vector = (weighted / (weights.sum()+1e-9)).reshape(1,-1)

        # Similitud coseno por generos
        movies_vectors = self.movies[self.genres_dummies.columns].values
        cos_sim = cosine_similarity(user_genre_vector, movies_vectors)[0]

        # DP: Alineamiento entre secuencia de generos
        dp_scores = []
        user_genre_sequence = self._user_genre_sequence(user_id)
        for idx, row in self.movies.iterrows():
            movie_seq = self._movie_genre_sequence(row)
            dp_score = needleman_wunsch_score(user_genre_sequence, movie_seq)
            dp_scores.append(dp_score)

        # Score combinado
        cos_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-9)
        dp_arr = np.array(dp_scores)
        dp_norm = (dp_arr - dp_arr.min()) / (dp_arr.max() - dp_arr.min() + 1e-9)

        # Rating influence
        rating_arr = self.movies["avg_rating"].values
        rating_norm = (rating_arr - rating_arr.min()) / (rating_arr.max() - rating_arr.min() + 1e-9)

        # Grafo proximity
        graph_scores = np.array([ self.grafo.proximity_score(user_id, mid) for mid in self.movies["movieId"].values ])
        graph_norm = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min() + 1e-9)

        combined_score = 0.4 * cos_norm + 0.25 * dp_norm + 0.25 * rating_norm + 0.1 * graph_norm

        # Attach score
        df = self.movies.copy()
        df["score"] = combined_score
        df["already_seen"] = df["movieId"].isin(peliculas_vistas)

        # Filtros
        df = df[~df["already_seen"]]
        if filtro_genero:
            df = df[df["genres"].str.contains(filtro_genero)]
        if min_rating and min_rating > 0:
            df = df[df["avg_rating"] >= min_rating]
        if year:
            try:
                df = df[df["title"].str.contains(str(year))]
            except:
                pass

        # Ordenamos usando quicksort
        rows = df.to_dict(orient="records")
        sorted_rows = quicksort_by_score(rows, key="score", reverse=True)

        # Posters
        top = sorted_rows[:n]
        for item in top:
            item["poster_url"] = self.poster_fetcher.get_poster_url(item["title"])

        # Resultado final
        result = [{
            "movieId": item["movieId"],
            "title": item["title"],
            "score": round(item["score"], 4),
            "avg_rating": round(item.get("avg_rating",0),2),
            "genres": item.get("genres",""),
            "poster_url": item.get("poster_url")
        } for item in top]

        return result

    def _user_genre_sequence(self, user_id):
        # Secuencia de géneros ordenada por preferencia del usuario
        ur = self.ratings[self.ratings["userId"] == user_id].merge(self.movies, on="movieId")
        if ur.empty:
            return []
        genre_cols = self.genres_dummies.columns
        weighted = (ur[genre_cols].multiply(ur["rating"], axis=0)).sum().sort_values(ascending=False)
        seq = [g for g,v in weighted.items() if v>0]
        return seq

    def _movie_genre_sequence(self, movie_row):
        return [g for g in self.genres_dummies.columns if movie_row.get(g,0)==1]