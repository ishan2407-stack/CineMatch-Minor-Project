import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split as surprise_train_test_split


print("\n==============================")
print("LOADING MOVIELENS DATASET")
print("==============================")

movies = pd.read_csv("Dataset/movies.csv")
ratings = pd.read_csv("Dataset/ratings.csv")

print("\nMovies Dataset:")
print(movies.head())

print("\nRatings Dataset:")
print(ratings.head())


print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("\nNumber of movies:", len(movies))
print("Number of ratings:", len(ratings))
print("Number of users:", ratings["userId"].nunique())

print("\nMovie columns:")
print(movies.columns)

print("\nRating columns:")
print(ratings.columns)


print("\n==============================")
print("MISSING VALUES")
print("==============================")

print("\nMovies missing values:")
print(movies.isnull().sum())

print("\nRatings missing values:")
print(ratings.isnull().sum())


print("\n==============================")
print("EXPLORATORY DATA ANALYSIS")
print("==============================")


plt.figure(figsize=(8, 5))

sns.countplot(
    x="rating",
    data=ratings
)

plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.savefig("Output/rating_distribution.png")

plt.show()


movie_rating_count = (
    ratings.groupby("movieId")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

plt.plot(movie_rating_count.values)

plt.title("Long Tail Distribution of Movie Popularity")
plt.xlabel("Movies sorted by number of ratings")
plt.ylabel("Number of ratings")

plt.savefig("Output/popularity_distribution.png")

plt.show()


top_movies = (
    ratings.groupby("movieId")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

top_movies = top_movies.reset_index(
    name="rating_count"
)

top_movies = top_movies.merge(
    movies,
    on="movieId"
)

print("\nTop 10 Most Rated Movies:")

print(
    top_movies[
        ["movieId", "title", "rating_count"]
    ]
)


print("\n==============================")
print("CREATING UTILITY MATRIX")
print("==============================")


utility_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("\nUtility Matrix:")
print(utility_matrix.head())

print("\nUtility Matrix Shape:")
print(utility_matrix.shape)


print("\n==============================")
print("SPARSITY CALCULATION")
print("==============================")


total_cells = (
    utility_matrix.shape[0]
    * utility_matrix.shape[1]
)

filled_cells = utility_matrix.count().sum()

empty_cells = total_cells - filled_cells

sparsity_ratio = (
    empty_cells / total_cells
) * 100

print("Total cells:", total_cells)
print("Filled cells:", filled_cells)
print("Empty cells:", empty_cells)

print(
    f"Sparsity Ratio: {sparsity_ratio:.2f}%"
)


print("\n==============================")
print("COSINE SIMILARITY")
print("==============================")


def cosine_similarity_numpy(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    dot_product = np.dot(
        vector_a,
        vector_b
    )

    magnitude_a = np.linalg.norm(
        vector_a
    )

    magnitude_b = np.linalg.norm(
        vector_b
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    similarity = (
        dot_product
        / (magnitude_a * magnitude_b)
    )

    return similarity


vector1 = [5, 4, 5]
vector2 = [5, 5, 4]

similarity = cosine_similarity_numpy(
    vector1,
    vector2
)

print(
    "Example Cosine Similarity:",
    similarity
)


print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================")


train_ratings, test_ratings = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42
)

print(
    "Training ratings:",
    len(train_ratings)
)

print(
    "Testing ratings:",
    len(test_ratings)
)


train_matrix = train_ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("\nTraining Matrix Shape:")
print(train_matrix.shape)


print("\n==============================")
print("USER-BASED COLLABORATIVE FILTERING")
print("==============================")


def predict_user_based(
    user_id,
    movie_id,
    train_matrix,
    k=20
):

    if user_id not in train_matrix.index:
        return train_ratings["rating"].mean()

    if movie_id not in train_matrix.columns:
        return train_ratings["rating"].mean()

    target_user = train_matrix.loc[user_id]

    similarities = []

    for other_user_id in train_matrix.index:

        if other_user_id == user_id:
            continue

        other_user = train_matrix.loc[
            other_user_id
        ]

        common_movies = (
            target_user.notna()
            & other_user.notna()
        )

        if common_movies.sum() == 0:
            continue

        vector_a = target_user[
            common_movies
        ].values

        vector_b = other_user[
            common_movies
        ].values

        similarity = cosine_similarity_numpy(
            vector_a,
            vector_b
        )

        similarities.append(
            (other_user_id, similarity)
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_users = similarities[:k]

    weighted_sum = 0
    similarity_sum = 0

    for other_user_id, similarity in top_users:

        rating = train_matrix.loc[
            other_user_id,
            movie_id
        ]

        if pd.isna(rating):
            continue

        weighted_sum += (
            similarity * rating
        )

        similarity_sum += abs(
            similarity
        )

    if similarity_sum == 0:
        return train_ratings["rating"].mean()

    predicted_rating = (
        weighted_sum / similarity_sum
    )

    predicted_rating = max(
        0.5,
        min(5.0, predicted_rating)
    )

    return predicted_rating


print("\n==============================")
print("ITEM-BASED COLLABORATIVE FILTERING")
print("==============================")


def predict_item_based(
    user_id,
    movie_id,
    train_matrix,
    k=20
):

    if user_id not in train_matrix.index:
        return train_ratings["rating"].mean()

    if movie_id not in train_matrix.columns:
        return train_ratings["rating"].mean()

    target_movie = train_matrix[
        movie_id
    ]

    similarities = []

    for other_movie_id in train_matrix.columns:

        if other_movie_id == movie_id:
            continue

        other_movie = train_matrix[
            other_movie_id
        ]

        common_users = (
            target_movie.notna()
            & other_movie.notna()
        )

        if common_users.sum() == 0:
            continue

        vector_a = target_movie[
            common_users
        ].values

        vector_b = other_movie[
            common_users
        ].values

        similarity = cosine_similarity_numpy(
            vector_a,
            vector_b
        )

        similarities.append(
            (other_movie_id, similarity)
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_movies = similarities[:k]

    weighted_sum = 0
    similarity_sum = 0

    for other_movie_id, similarity in top_movies:

        rating = train_matrix.loc[
            user_id,
            other_movie_id
        ]

        if pd.isna(rating):
            continue

        weighted_sum += (
            similarity * rating
        )

        similarity_sum += abs(
            similarity
        )

    if similarity_sum == 0:
        return train_ratings["rating"].mean()

    predicted_rating = (
        weighted_sum / similarity_sum
    )

    predicted_rating = max(
        0.5,
        min(5.0, predicted_rating)
    )

    return predicted_rating


print("\n==============================")
print("TEST USER-BASED PREDICTION")
print("==============================")


sample_user = int(
    test_ratings.iloc[0]["userId"]
)

sample_movie = int(
    test_ratings.iloc[0]["movieId"]
)

user_prediction = predict_user_based(
    sample_user,
    sample_movie,
    train_matrix
)

print(
    "User ID:",
    sample_user
)

print(
    "Movie ID:",
    sample_movie
)

print(
    "Predicted Rating:",
    round(
        user_prediction,
        2
    )
)


print("\n==============================")
print("TEST ITEM-BASED PREDICTION")
print("==============================")


item_prediction = predict_item_based(
    sample_user,
    sample_movie,
    train_matrix
)

print(
    "Predicted Rating:",
    round(
        item_prediction,
        2
    )
)


print("\n==============================")
print("USER-BASED RMSE")
print("==============================")


evaluation_data = test_ratings.head(1000)

actual_user = []
predicted_user = []

for _, row in evaluation_data.iterrows():

    user_id = int(row["userId"])
    movie_id = int(row["movieId"])
    actual_rating = row["rating"]

    prediction = predict_user_based(
        user_id,
        movie_id,
        train_matrix,
        k=20
    )

    actual_user.append(
        actual_rating
    )

    predicted_user.append(
        prediction
    )


user_rmse = np.sqrt(
    mean_squared_error(
        actual_user,
        predicted_user
    )
)

print(
    "User-Based RMSE:",
    round(user_rmse, 4)
)


print("\n==============================")
print("ITEM-BASED RMSE")
print("==============================")


actual_item = []
predicted_item = []

evaluation_data_item = test_ratings.head(300)

for _, row in evaluation_data_item.iterrows():

    user_id = int(row["userId"])
    movie_id = int(row["movieId"])
    actual_rating = row["rating"]

    prediction = predict_item_based(
        user_id,
        movie_id,
        train_matrix,
        k=20
    )

    actual_item.append(
        actual_rating
    )

    predicted_item.append(
        prediction
    )


item_rmse = np.sqrt(
    mean_squared_error(
        actual_item,
        predicted_item
    )
)

print(
    "Item-Based RMSE:",
    round(item_rmse, 4)
)


print("\n==============================")
print("SVD MODEL")
print("==============================")


reader = Reader(
    rating_scale=(0.5, 5)
)

data = Dataset.load_from_df(
    ratings[
        [
            "userId",
            "movieId",
            "rating"
        ]
    ],
    reader
)

trainset, testset = surprise_train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

svd_model = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42
)

print("Training SVD model...")

svd_model.fit(
    trainset
)

print("SVD training completed!")


print("\n==============================")
print("SVD PREDICTIONS")
print("==============================")


svd_predictions = svd_model.test(
    testset
)

actual_svd = [
    prediction.r_ui
    for prediction in svd_predictions
]

predicted_svd = [
    prediction.est
    for prediction in svd_predictions
]

svd_rmse = np.sqrt(
    mean_squared_error(
        actual_svd,
        predicted_svd
    )
)

print(
    "SVD RMSE:",
    round(svd_rmse, 4)
)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")


comparison = pd.DataFrame(
    {
        "Model": [
            "User-Based",
            "Item-Based",
            "SVD"
        ],
        "RMSE": [
            user_rmse,
            item_rmse,
            svd_rmse
        ]
    }
)

print(
    comparison
)


plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    x="Model",
    y="RMSE",
    data=comparison
)

plt.title(
    "RMSE Comparison of Recommendation Models"
)

plt.xlabel("Model")
plt.ylabel("RMSE")

plt.savefig("Output/rmse_comparison.png")

plt.show()


print("\n==============================")
print("MOVIE RECOMMENDATION SYSTEM")
print("==============================")


def recommendmovies(
    userid,
    n=5
):

    if userid not in ratings["userId"].values:

        print(
            "User ID not found."
        )

        return

    watched_movies = set(
        ratings[
            ratings["userId"] == userid
        ]["movieId"]
    )

    movies_to_predict = movies[
        ~movies["movieId"].isin(
            watched_movies
        )
    ].copy()

    predictions = []

    for movie_id in movies_to_predict[
        "movieId"
    ]:

        prediction = svd_model.predict(
            userid,
            movie_id
        )

        predictions.append(
            prediction.est
        )

    movies_to_predict[
        "predicted_rating"
    ] = predictions

    recommendations = (
        movies_to_predict
        .sort_values(
            "predicted_rating",
            ascending=False
        )
        .head(n)
    )

    print(
        f"\nTop {n} Movie Recommendations "
        f"for User {userid}:"
    )

    for index, row in recommendations.iterrows():

        print(
            f"{row['title']} "
            f"--> Predicted Rating: "
            f"{row['predicted_rating']:.2f}"
        )

    return recommendations


print("\n==============================")
print("FINAL DEMO")
print("==============================")


recommendations = recommendmovies(
    userid=1,
    n=5
)


if recommendations is not None:

    with open(
        "Output/recommendation_output.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CineMatch - Movie Recommendations\n\n"
        )

        file.write(
            "User ID: 1\n\n"
        )

        for index, row in recommendations.iterrows():

            file.write(
                f"{row['title']} --> "
                f"Predicted Rating: "
                f"{row['predicted_rating']:.2f}\n"
            )


print("\n==============================")
print("CINEMATCH PROJECT COMPLETED")
print("==============================")