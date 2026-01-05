import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_data.csv")
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Fill NaN values
    df['Director'] = df['Director'].fillna("Unknown")
    df['Cast'] = df['Cast'].fillna("Unknown")
    df['Description'] = df['Description'].fillna("")
    df['Type'] = df['Type'].fillna("")
    df.dropna(subset=["Country", "Release_Date"], inplace=True)
    # Convert release date
    df['Release_Date'] = pd.to_datetime(df['Release_Date'].str.strip(), errors="coerce")
    df['Release_Year'] = df['Release_Date'].dt.year
    df['Release_Month'] = df['Release_Date'].dt.month
    return df

data = load_data()

# ------------------------
# Sidebar navigation
# ------------------------
st.sidebar.title("📺 Netflix Data App")
page = st.sidebar.radio("Go to", ["Visualizations", "Recommendation System"])

# ------------------------
# Page 1: Visualizations
# ------------------------
if page == "Visualizations":
    st.title("📊 Netflix Visualizations")

    # --- Filters ---
    st.sidebar.subheader("🔎 Filters")
    year_filter = st.sidebar.selectbox("Select Year", ["All"] + sorted(data['Release_Year'].dropna().unique().tolist()))
    country_filter = st.sidebar.selectbox("Select Country", ["All"] + sorted(data['Country'].dropna().unique().tolist()))
    category_filter = st.sidebar.selectbox("Select Category", ["All"] + data['Category'].dropna().unique().tolist())

    filtered_data = data.copy()
    if year_filter != "All":
        filtered_data = filtered_data[filtered_data['Release_Year'] == year_filter]
    if country_filter != "All":
        filtered_data = filtered_data[filtered_data['Country'].str.contains(country_filter, na=False)]
    if category_filter != "All":
        filtered_data = filtered_data[filtered_data['Category'] == category_filter]

    st.write(f"### Showing {filtered_data.shape[0]} titles after filtering")

    # 1. Movies vs TV Shows
    st.subheader("Count of Movies vs TV Shows")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_data, x="Category", ax=ax)
    st.pyplot(fig)

    # 2. Top 10 countries
    st.subheader("Top 10 Countries by Content")
    if not filtered_data.empty:
        top_countries = filtered_data['Country'].str.split(", ", expand=True).stack().value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data available for selected filters.")

    # 3. Content released by year
    st.subheader("Content Released by Year")
    if not filtered_data.empty:
        content_by_year = filtered_data['Release_Year'].value_counts().sort_index()
        fig, ax = plt.subplots()
        content_by_year.plot(kind="line", marker="o", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data available for selected filters.")

elif page == "Recommendation System":
    st.title("Netflix Recommendation System")
    st.write("Get content recommendations based on description, cast, and director similarity.")

    # Build TF-IDF once
    @st.cache_resource
    def build_recommender(data):
        data['content_info'] = data['Description'] + " " + data['Director'] + " " + data['Cast'] + " " + data['Type']
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(data['content_info'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(data.index, index=data['Title']).drop_duplicates()
        return cosine_sim, indices

    cosine_sim, indices = build_recommender(data)

    def recommend_movies(title, n=5):
        idx = indices.get(title)
        if idx is None:
            return pd.DataFrame()
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return data.iloc[movie_indices]

    # --- UI elements ---
    movie_name = st.selectbox("Select a movie/TV show title:", sorted(data['Title'].unique()))
    num_recs = st.slider("Number of recommendations:", 3, 15, 5)

    if st.button("Get Recommendations"):
        recs = recommend_movies(movie_name, n=num_recs)
        if not recs.empty:
            st.subheader(f"Recommended titles similar to **{movie_name}**:")
            # Display in grid (cards)
            for i in range(0, recs.shape[0], 3):  # 3 per row
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < recs.shape[0]:
                        row = recs.iloc[i + j]
                        with col:
                            st.markdown(
                                f"""
                                <div style="padding:10px; border-radius:10px; background:#f8f8f8;
                                            margin-bottom:10px; text-align:center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                    <h4 style="color:#e50914; margin-bottom:5px;">{row['Title']}</h4>
                                    <p><b>Category:</b> {row['Category']}</p>
                                    <p><b>Year:</b> {row['Release_Year']}</p>
                                    <p style="font-size:12px; color:#555;">{row['Description'][:120]}...</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        else:
            st.error("Title not found in dataset. Please try another.")
