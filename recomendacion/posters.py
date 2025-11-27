# posters.py
import os
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

class PosterFetcher:
    def __init__(self, api_key=TMDB_API_KEY):
        self.api_key = api_key
        self.base_search = "https://api.themoviedb.org/3/search/movie"
        self.base_image = "https://image.tmdb.org/t/p/w342"  # tamaño medio

    def get_poster_url(self, title):
        if not self.api_key:
            return None
        try:
            params = {
                "api_key": self.api_key,
                "query": title,
                "include_adult": False,
                "language": "en-US",
                "page": 1
            }
            r = requests.get(self.base_search, params=params, timeout=5)
            data = r.json()
            results = data.get("results")
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    return self.base_image + poster_path
        except Exception as e:
            # error silencioso para no romper la app
            pass
        return None
