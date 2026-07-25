import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('movies_metadata.csv')

# Basic exploration
df.head()
df.columns
df.info()
df.shape
df.describe()
df.isnull().sum()

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Select required columns
df = df[['title', 'overview', 'genres', 'tagline', 'vote_average', 'popularity']]

df

# Remove missing titles
df = df.dropna(subset=['title'])

# Fill missing values
df['overview'] = df['overview'].fillna('')

# Convert genres column
import ast

df.iloc[0]['genres']
ast.literal_eval(df.iloc[0]['genres'])

df['genres'] = df['genres'].apply(
    lambda x: " ".join([i['name'] for i in ast.literal_eval(x)])
)

df.head()

df['tagline'] = df['tagline'].fillna('')

df.isnull().sum()

# Create tags column
df['tags'] = df['overview'] + " " + df['genres'] + " " + df['tagline']

df
df['tags'][1]

# Remove stop words, punctuation and perform lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lower case
    text = text.lower()

    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Tokenization
    words = text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply preprocessing
df['tags'] = df['tags'].apply(preprocess_text)

df['tags'][1]

# Reset index
df = df.reset_index(drop=True)

# Create title-index mapping
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

indices

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words='english'
)

# Create TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['tags'])

tfidf_matrix

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(title, n=10):
    title = title.lower()

    if title not in indices:
        return ['movie not found']

    idx = indices[title]

    # Calculate similarity scores
    sim_score = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Get top similar movies
    similar_idx = sim_score.argsort()[::-1][1:n+1]

    return df['title'].iloc[similar_idx]

# Example
get_recommendations('avenger')

# Save files
import pickle

pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
pickle.dump(indices, open('indices.pkl', 'wb'))
df.to_pickle('df.pkl')
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))