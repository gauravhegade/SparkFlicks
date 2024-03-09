from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

ratings = pd.read_csv("input/ratings.csv")
movies = pd.read_csv("input/movies.csv")
unique_user = ratings.userId.nunique(dropna=True)
unique_movie = ratings.movieId.nunique(dropna=True)

total_ratings = unique_user * unique_movie
rating_present = ratings.shape[0]

ratings_not_provided = total_ratings - rating_present
sparsity = ratings_not_provided / total_ratings

movie_freq = pd.DataFrame(ratings.groupby("movieId").size(), columns=["count"])

threshold_rating_freq = 10

popular_movies_id = list(
    set(movie_freq.query("count>=@threshold_rating_freq").index))

ratings_with_popular_movies = ratings[ratings.movieId.isin(popular_movies_id)]

user_cnt = pd.DataFrame(ratings.groupby("userId").size(), columns=["count"])

threshold_val = 30
active_user = list(set(user_cnt.query("count>=@threshold_val").index))

ratings_with_popular_movies_with_active_user = ratings_with_popular_movies[
    ratings_with_popular_movies.userId.isin(active_user)
]

final_ratings = ratings_with_popular_movies_with_active_user
item_user_mat = final_ratings.pivot(
    index="movieId", columns="userId", values="rating"
).fillna(0)

# create a mapper which maps movie index and its title
movie_to_index = {
    movie: i
    for i, movie in enumerate(
        list(movies.set_index("movieId").loc[item_user_mat.index].title)
    )
}

# create a sparse matrix for more efficient calculations
item_user_mat_sparse = csr_matrix(item_user_mat.values)


# fuzzy_movie_name_matching


def fuzzy_movie_name_matching(input_str, mapper, print_matches):
    # match_movie is list of tuple of 3 values(movie_name,index,fuzz_ratio)
    match_movie = []
    for movie, ind in mapper.items():
        current_ratio = fuzz.ratio(movie.lower(), input_str.lower())
        if current_ratio >= 50:
            match_movie.append((movie, ind, current_ratio))

    # sort the match_movie with respect to ratio

    match_movie = sorted(match_movie, key=lambda x: x[2])[::-1]

    if len(match_movie) == 0:
        print("No such movie exists in database.\n")
        return -1
    if print_matches == True:
        print(f"Some matches of {input_str} are: \n")
        for title, ind, ratio in match_movie:
            print(title, ind, "\n")

    return match_movie[0][1]


# define the model
recommendation_model = NearestNeighbors(
    metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1
)


# create a function which takes a movie name and make recommedation for it


def make_recommendation(input_str, n_recommendation):
    data = item_user_mat_sparse
    model = recommendation_model
    mapper = movie_to_index
    res = []

    model.fit(data)

    index = fuzzy_movie_name_matching(input_str, mapper, print_matches=False)

    if index == -1:
        print("pls enter a valid movie name\n")
        return

    index_list = model.kneighbors(
        data[index], n_neighbors=n_recommendation + 1, return_distance=False
    )
    # now we ind of all recommendation
    # build mapper index->title
    index_to_movie = {ind: movie for movie, ind in mapper.items()}

    for i in range(1, index_list.shape[1]):
        res.append(index_to_movie[index_list[0][i]])

    return res
