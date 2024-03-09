from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import recommender
from dotenv import load_dotenv

load_dotenv()
custom_search_api = os.getenv('CUSTOM_SEARCH_API')
custom_search_code = os.getenv('CUSTOM_SEARCH_CODE')

app = Flask("SparkFlicks")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_movie = request.form.get("input_movie").lower()
        number_of_recommendations = request.form.get(
            "number_of_recommendations")
        number_of_recommendations = int(number_of_recommendations)

        recommender_output = recommender.make_recommendation(
            input_str=input_movie, n_recommendation=number_of_recommendations)


        return render_template("home.html", recommender_output=recommender_output, n_rec=number_of_recommendations, input_movie=input_movie, enumerate=enumerate, key = custom_search_api, cx = custom_search_code)

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
