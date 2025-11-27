# grafos.py
from collections import defaultdict, deque

class UsuarioPeliculaGrafo:
    def __init__(self, ratings_df):
        # adjacency: usuario -> set(movieId), movie -> set(userId)
        self.user_to_movies = defaultdict(set)
        self.movie_to_users = defaultdict(set)
        for _, row in ratings_df.iterrows():
            u = int(row["userId"])
            m = int(row["movieId"])
            self.user_to_movies[u].add(m)
            self.movie_to_users[m].add(u)

    def users_who_rated(self, movie_id):
        return self.movie_to_users.get(movie_id, set())

    def movies_rated_by(self, user_id):
        return self.user_to_movies.get(user_id, set())

    def bfs_user_to_movie_distance(self, start_user, target_movie, max_depth=3):
        """
        bfs en bipartite grafo usuario<->pelicula:
        niveles: user -> movie -> user -> movie ...
        retornamos la distancia en pasos (si no encontrado, return inf)
        """
        visited_users = set()
        visited_movies = set()
        q = deque()
        q.append(("u", start_user, 0))
        visited_users.add(start_user)

        while q:
            typ, node, depth = q.popleft()
            if typ == "u":
                # expandir a películas
                for m in self.user_to_movies.get(node, []):
                    if m == target_movie:
                        return depth + 1
                    if m not in visited_movies and depth+1 <= max_depth:
                        visited_movies.add(m)
                        q.append(("m", m, depth+1))
            else:
                # typ == "m"
                for u in self.movie_to_users.get(node, []):
                    if u not in visited_users and depth+1 <= max_depth:
                        visited_users.add(u)
                        q.append(("u", u, depth+1))
        return float("inf")

    def proximity_score(self, user_id, movie_id):
        # score inverso a la distancia: más cercano => mayor score
        d = self.bfs_user_to_movie_distance(user_id, movie_id, max_depth=4)
        if d == float("inf"):
            return 0.0
        # transformamos distancia a score: 1/(1+d)
        return 1.0 / (1.0 + d)
