from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import faiss
import joblib

from sentence_transformers import SentenceTransformer

from src.semantic_cache import SemanticCache


app = FastAPI()

print("Loading system...")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("models/faiss_index.bin")

# Load dataset
df = pd.read_csv("data/cleaned_newsgroups.csv")

documents = df["clean_text"].tolist()
labels = df["label"].tolist()

# Load clustering model
gmm = joblib.load("models/gmm_cluster_model.pkl")

# Initialize semantic cache
cache = SemanticCache()

print("System ready!")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(request: QueryRequest):

    query = request.query

    # Generate embedding
    query_embedding = model.encode(query)

    # Check semantic cache
    hit, entry, score = cache.lookup(query_embedding)

    if hit:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(score),
            "results": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

    # Search FAISS vector database
    D, I = index.search(query_embedding.reshape(1, -1), 5)

    results = []

    for rank, idx in enumerate(I[0]):

        snippet = documents[idx][:200]

        results.append({
            "rank": rank + 1,
            "snippet": snippet,
            "label": labels[idx],
            "similarity_score": float(D[0][rank])
        })

    # Predict dominant cluster
    cluster_probs = gmm.predict_proba([query_embedding])
    dominant_cluster = int(cluster_probs.argmax())

    # Store in cache
    cache.add(query, query_embedding, results, dominant_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "results": results,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}