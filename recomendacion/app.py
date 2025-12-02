# app.py
from flask import Flask, render_template, request
from recomendador import Recomendador
import os

app = Flask(__name__, static_folder='../static')
reco = Recomendador(data_folder="data")

# obtener lista de usuarios
usuarios_completos = sorted(reco.ratings["userId"].unique())
usuarios = usuarios_completos[:10] # <-- Límite a los 5 primeros usuarios

@app.route("/")
def index():
    # generos disponibles para filtro
    generos = reco.get_all_genres()
    return render_template("index.html", usuarios=usuarios, generos=generos)

@app.route("/recomendar", methods=["POST"])
def recomendar():
    user = int(request.form["user"])
    filtro_genero = request.form.get("genero") or None
    min_rating = float(request.form.get("min_rating") or 0)
    year = request.form.get("year") or None

    # obteniene recomendaciones 
    recomendaciones = reco.recomendar_peliculas(
        user_id=user,
        n=10,
        filtro_genero=filtro_genero,
        min_rating=min_rating,
        year=year
    )

    # obteniene calificaciones del usuario
    calificaciones = reco.get_user_ratings(user)

    return render_template(
        "recomendaciones.html",
        user=user,
        recs=recomendaciones,
        ratings=calificaciones
    )

if __name__ == "__main__":
    app.run(debug=True)