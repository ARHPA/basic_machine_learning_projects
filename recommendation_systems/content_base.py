import numpy as np
import pandas as pd

movie_df = pd.read_csv("/home/amirreza/Desktop/basic_machine_learning_projects/recommendation_systems/movies.csv")
rating_df = pd.read_csv("/home/amirreza/Desktop/basic_machine_learning_projects/recommendation_systems/ratings.csv")

movie_df["year"] = movie_df["title"].str.extract("(\(\d\d\d\d\))", expand=False)
movie_df["year"] = movie_df["year"].str.extract("(\d\d\d\d)", expand=False)
#movie_df["year"] = pd.to_numeric(movie_df["year"])
movie_df = movie_df.dropna()
#movie_df["year"] = movie_df["year"].astype(int)
movie_df["title"] = movie_df["title"].str.replace("(\(\d\d\d\d\))", "")
movie_df["title"] = movie_df["title"].apply(lambda x: x.strip())
movie_df["genres"] = movie_df["genres"].apply(lambda x: x.split("|"))
first_movie_df = movie_df.copy()

for index, row in movie_df.iterrows():
    for genres in row["genres"]:
        movie_df.at[index, genres] = 1
    #print(row["year"])
    if int(row["year"]) >= 2000:
        movie_df.at[index, ">2000"] = 0.5
    if int(row["year"]) >= 2010:
        movie_df.at[index, ">2000"] = 1
movie_df = movie_df.fillna(0)


rating_df["timestamp"] = rating_df["timestamp"] > 946080000

userInput = [
            {'title':'Iron Man', 'rating':4.5},
            {'title':'Interstellar', 'rating':5},
            {'title':'Captain America: The First Avenger', 'rating':4.5},
            {'title':'Iron Man 3', 'rating':4},
            {'title':'Thor', 'rating':4},
            {'title':"Pulp Fiction", 'rating':3.5},
         ] 

input_df = pd.DataFrame(userInput)

#print(input_df)

user_input_movie = first_movie_df[movie_df["title"].isin(input_df["title"].tolist())]
input_movie = movie_df[movie_df["title"].isin(input_df["title"].tolist())]
input_movie = pd.merge(input_df, input_movie).drop("movieId", axis=1).drop("title", axis=1).drop("genres", axis=1).drop("(no genres listed)", axis=1).drop("year", axis=1)
movie_matrix = movie_df.drop("title", axis=1).drop("genres", axis=1).drop("(no genres listed)", axis=1).drop("year", axis=1)

user_score = input_movie["rating"]
user_genres = input_movie.drop("rating", axis=1)

genres_score = user_score.transpose().dot(user_genres)
movie_matrix = movie_matrix.set_index(movie_matrix["movieId"]).drop("movieId", 1)
recommendation_df = (genres_score * movie_matrix).sum(axis=1) / genres_score.sum() * 100
recommendation_df = recommendation_df.sort_values(ascending=False)
recommendation_df = first_movie_df[~first_movie_df["movieId"].isin(user_input_movie["movieId"])].loc[first_movie_df["movieId"].isin(recommendation_df.head(20).keys())]
#recommendation_df = recommendation_df[~recommendation_df["movieId"].isin(input_movie["movieId"])]

print("\n\n20 best movies you should to see:")
print(recommendation_df[["movieId", "title", "year"]])
