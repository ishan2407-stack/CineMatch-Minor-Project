import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="centered")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

@st.cache_resource
def train_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)
    return model

movies, ratings = load_data()
model = train_model(ratings)

st.title("🎬 CineMatch")
st.subheader("Movie Recommendation System")
st.write("Personalized movie recommendations using Collaborative Filtering and SVD.")

user_ids = sorted(ratings["userId"].unique().tolist())
user_id = st.selectbox("Select User ID", user_ids, index=0)

if st.button("🍿 Recommend Movies", use_container_width=True):
    rated = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    candidates = movies[~movies["movieId"].isin(rated)].copy()

    candidates["predicted_rating"] = candidates["movieId"].apply(
        lambda mid: model.predict(user_id, int(mid)).est
    )

    top5 = candidates.sort_values("predicted_rating", ascending=False).head(5)

    st.success(f"Top 5 recommendations for User ID {user_id}")

    for i, (_, row) in enumerate(top5.iterrows(), start=1):
        st.markdown(
            f"### {i}. {row['title']}\n"
            f"**Predicted Rating:** {row['predicted_rating']:.2f} ⭐"
        )

st.divider()
st.caption("CineMatch | B.Tech CSE (AI & ML) Minor Project")
