# posters.py
import os
import requests
import re  # <--- IMPORTANTE: Necesario para limpiar el título
from dotenv import load_dotenv

# Cargar variables de entorno
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
        
        # Esta linea borra el " (1995)" para buscar solo "Jumanji"
        clean_title = re.sub(r'\s*\(\d{4}\)$', '', title)
        
        # Mensaje en terminal para que se vea que esta pasando
        print(f"Buscando: '{clean_title}' ...", end=" ")

        try:
            params = {
                "api_key": self.api_key,
                "query": clean_title,
                "include_adult": "false",
                "language": "en-US", 
                "page": 1
            }
            # Timeout de 3 segundos para que no se congele si internet falla
            r = requests.get(self.base_search, params=params, timeout=3)
            
            if r.status_code == 200:
                data = r.json()
                results = data.get("results")
                if results:
                    poster_path = results[0].get("poster_path")
                    if poster_path:
                        print("¡Encontrado!")
                        return self.base_image + poster_path
                print("Sin resultados")
            else:
                print(f"Error API: {r.status_code}")

        except Exception as e:
            print(f"Excepción: {e}")
            pass
        
        return None