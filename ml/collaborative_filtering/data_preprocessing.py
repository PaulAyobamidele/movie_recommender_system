import pandas as pd
from sklearn.model_selection import train_test_split

ratings_df = pd.read_csv("../../../movielens/rating.csv")
movies_df = pd.read_csv("../../../movielens/movie.csv")
link_df = pd.read_csv("../../../movielens/link.csv")
genome_scores_df = pd.read_csv("../../../movielens/genome_scores.csv")
genome_tags_df = pd.read_csv("../../../movielens/genome_tags.csv")
tags_df = pd.read_csv("../../../movielens/tag.csv")


print("Ratings dataset shape: ", ratings_df.shape)
print("Movies dataset shape: ", movies_df.shape)


merged_ratings_movies_df = pd.merge(ratings_df, movies_df, on="movieId")

train_data, test_data = train_test_split(
    merged_ratings_movies_df, test_size=0.2, random_state=42
)


# Normalize ratings

min_rating = merged_ratings_movies_df["rating"].min()
max_rating = merged_ratings_movies_df["rating"].max()


def normalize_ratings(rating):
    return (rating - min_rating) / (max_rating - min_rating)


train_data["rating_normalized"] = normalize_ratings(train_data["rating"])
test_data["rating_normalized"] = normalize_ratings(test_data["rating"])


print(train_data.isnull().sum())
print(test_data.isnull().sum())
