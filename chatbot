# Next Level Semantic Search (with scoring)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset
data = [
    "Artificial Intelligence is the simulation of human intelligence",
    "Machine Learning is a subset of AI that learns from data",
    "Deep Learning uses neural networks with multiple layers",
    "Natural Language Processing helps computers understand text"
]

# Convert to vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data)

def search(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectors)

    best_match_index = similarity.argmax()
    score = similarity[0][best_match_index]

    return data[best_match_index], score

# Chat loop
while True:
    query = input("Ask: ")
    answer, score = search(query)

    print(f"Answer: {answer}")
    print(f"Confidence Score: {round(score, 2)}\n")
