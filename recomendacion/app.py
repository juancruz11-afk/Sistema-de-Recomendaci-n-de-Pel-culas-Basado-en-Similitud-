# app.py
from flask import Flask, render_template, request
from recomendador import Recomendador
import os

app = Flask(__name__)
reco = Recomendador(data_folder="data")

# obtener lista de usuarios
usuarios = sorted(reco.ratings["userId"].unique())

@app.route("/")
def index():
    # géneros disponibles para filtro
    generos = reco.get_all_genres()
    return render_template("index.html", usuarios=usuarios, generos=generos)

@app.route("/recomendar", methods=["POST"])
def recomendar():
    user = int(request.form["user"])
    filtro_genero = request.form.get("genero") or None
    min_rating = float(request.form.get("min_rating") or 0)
    year = request.form.get("year") or None

    # obtener recomendaciones (top 10 internamente, luego mostramos 3 o las que pida)
    recomendaciones = reco.recomendar_peliculas(
        user_id=user,
        n=10,
        filtro_genero=filtro_genero,
        min_rating=min_rating,
        year=year
    )

    # obtener calificaciones del usuario
    calificaciones = reco.get_user_ratings(user)

    return render_template(
        "recomendaciones.html",
        user=user,
        recs=recomendaciones,
        ratings=calificaciones
    )

if __name__ == "__main__":
    app.run(debug=True)
