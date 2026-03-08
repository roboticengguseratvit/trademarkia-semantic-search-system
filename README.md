# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a **lightweight semantic search system** built on the **20 Newsgroups dataset (~20,000 documents across 20 categories)**.

The system demonstrates how modern information retrieval systems combine:

- Vector embeddings
- Vector databases
- Fuzzy clustering
- Semantic caching
- API-based query interfaces

The implementation includes **analysis notebooks, evaluation metrics, and a FastAPI service**.

---

# System Architecture


User Query
   ↓
Sentence Transformer Embedding
   ↓
Semantic Cache Lookup
   ↓
Vector Database Search (FAISS)
   ↓
Top-K Document Retrieval
   ↓
Fuzzy Cluster Prediction
   ↓
API Response


---

# Dataset

Dataset used:

**20 Newsgroups**

~20,000 documents across 20 discussion categories.

Example categories:

- sci.space
- comp.graphics
- rec.sport.hockey
- talk.politics.misc
- alt.atheism
- sci.med

Dataset characteristics:

- Informal discussions
- Overlapping topics
- Noisy formatting

This makes it a useful benchmark for **semantic search systems**.

---

# Project Structure

```
semantic-search-system
│
├── api
│   └── main.py
│
├── src
│   ├── semantic_cache.py
│   ├── search_engine.py
│   ├── vector_store.py
│   ├── clustering.py
│   └── embedding_model.py
│
├── notebooks
│   ├── 01_prepare_data.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_vector_database.ipynb
│   ├── 04_fuzzy_clustering.ipynb
│   ├── 05_cluster_visualisation.ipynb
│   ├── 06_performance_evaluation.ipynb
│   └── 07_cache_evaluation.ipynb
│
├── models
│   ├── faiss_index.bin
│   ├── document_embeddings.npy
│   └── gmm_cluster_model.pkl
│
├── data
│   └── cleaned_newsgroups.csv
│
├── requirements.txt
└── README.md
```

---

# Embedding Model

Model used:


all-MiniLM-L6-v2


Reasons for selection:

- Lightweight (384-dimensional embeddings)
- Fast inference
- Strong semantic similarity performance
- Widely used for semantic search systems

The model converts documents and queries into **dense vector representations**.

---

# Vector Database

Vector search implemented using:


FAISS IndexFlatL2


Advantages:

- Fast nearest-neighbour search
- Efficient for ~20k vectors
- Easy Python integration

FAISS enables **semantic similarity search instead of keyword matching**.

---

# Fuzzy Clustering

Although the dataset has 20 labeled categories, many documents overlap across topics.

Example:


Gun laws in politics discussion


This document may belong to both:


talk.politics.misc
talk.politics.guns


To capture this ambiguity, the system uses **Gaussian Mixture Models (GMM)**.

GMM produces **soft cluster memberships**.

Example:


Cluster probabilities:

Cluster 3 → 0.65
Cluster 7 → 0.25
Cluster 12 → 0.10


This approach better represents **topic overlap** compared to hard clustering.

---

# Cluster Analysis

Cluster visualization was performed using **PCA dimensionality reduction**.

The embedding space was reduced to two dimensions to visualize semantic grouping.

Key observations:

- Documents from similar topics form clusters.
- Technical discussions group together.
- Some documents lie between clusters, showing topic overlap.

Boundary documents were identified using **cluster probability confidence scores**, demonstrating the effectiveness of fuzzy clustering.

---

# Semantic Cache

Traditional caches rely on **exact string matching**.

Example:


"What is AI?"
"Explain artificial intelligence"


A traditional cache treats these as different queries.

This system implements **semantic caching** using:

- query embeddings
- cosine similarity

Cache lookup process:


query embedding
↓
compare with cached embeddings
↓
similarity threshold check


Similarity threshold used:


0.85


If similarity exceeds the threshold, cached results are reused.

---

# Cache Evaluation

Cache performance was evaluated using groups of **semantically similar queries**.

Example queries:


space shuttle mission
nasa spacecraft mission
space mission nasa
nasa shuttle launch


Results show that semantic caching can reuse results across differently phrased queries.

Cache statistics tracked:

- total_entries
- hit_count
- miss_count
- hit_rate

---

# Performance Evaluation

Performance experiments measured:

### Query Latency

Average response time:


~30–50 ms per query


---

### Precision@5

Semantic search quality was evaluated using labeled topics.

Example queries:


space shuttle launch → sci.space
hockey playoffs → rec.sport.hockey
graphics rendering → comp.graphics


Precision@5 measures whether the expected category appears in the top search results.

---

### Cache Hit Rate

Cache experiments show that semantic caching improves efficiency when similar queries repeat.

---

# API Endpoints

### Query API

```
POST /query
```

Example request:

```json
{
 "query": "space shuttle mission"
}
```

Example response:

```json
{
 "query": "space shuttle mission",
 "cache_hit": false,
 "results": [
   {
     "rank": 1,
     "snippet": "...",
     "label": "sci.space",
     "similarity_score": 1.23
   }
 ],
 "dominant_cluster": 19
}
```

Cache Statistics
GET /cache/stats

Example response:

{
 "total_entries": 4,
 "hit_count": 2,
 "miss_count": 2,
 "hit_rate": 0.5
}
Clear Cache
DELETE /cache

Resets the cache and statistics.

Quick Start

Clone the repository:

git clone <repository_url>

Navigate to the project folder:

cd semantic-search-system

Install dependencies:

pip install -r requirements.txt

Start the API server:

uvicorn api.main:app --reload

Open API documentation:

http://127.0.0.1:8000/docs
Notebooks

The repository includes several notebooks for experimentation and analysis.

Notebook	Purpose
01_prepare_data	Data preprocessing
02_embeddings	Generate document embeddings
03_vector_database	Build FAISS vector index
04_fuzzy_clustering	Train Gaussian Mixture clustering
05_cluster_visualisation	Visualize clusters
06_performance_evaluation	Evaluate search performance
07_cache_evaluation	Evaluate semantic cache
Key Contributions

This project demonstrates:

Semantic document retrieval using vector embeddings

Fuzzy clustering for overlapping topics

Semantic query caching

Vector database search using FAISS

API deployment using FastAPI

Performance and cache evaluation experiments

Future Improvements

Possible improvements include:

Cluster-aware cache lookup

Hybrid search (vector + keyword)

Larger embedding models

Scalable vector databases such as Milvus or Pinecone

# Author

Rithikha S (23BRS1374)

Implementation of a **Semantic Search System with Fuzzy Clustering and Semantic Caching** for the **Trademarkia AI/ML Engineer Task**.

GitHub: 