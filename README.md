# Clustering Documents with Partial Similarity

This project explores how to automatically cluster textual documents that exhibit partial or semantic similarity using modern NLP and unsupervised machine learning techniques.

## Project Overview

Given a collection of paraphrased or reworded documents, our goal is to group them into semantically coherent clusters — even when they share limited lexical overlap. We use SBERT embeddings and clustering algorithms to achieve this.

---

## Dataset

- **Source**: [PlagBench Dataset (2024)](https://github.com/Brit7777/plagbench)
- **Used field**: `susp_doc` — suspicious (potentially paraphrased) documents
- **Size**: ~3000 cleaned documents

Preprocessing steps included:
- Lowercasing
- Removing punctuation
- Deduplication and whitespace normalization

---

## Methodology

### 1. Text Embeddings
- **Model**: `all-MiniLM-L6-v2` from [Sentence-Transformers](https://www.sbert.net/)
- Each document converted into a 384-dimensional vector

### 2. Clustering Algorithms
- **KMeans** (baseline): evaluated using Silhouette Score
- **DBSCAN** (primary): detects clusters of arbitrary shape and filters out noise

### 3. Evaluation & Visualization
- **Silhouette Score** to assess cluster quality
- **t-SNE** to visualize document clusters in 2D

---

## Results

| Method   | Clusters Found | Silhouette Score |
|----------|----------------|------------------|
| KMeans   | 3              | 0.0525           |
| DBSCAN   | 40             | 0.136            |

DBSCAN provided more meaningful separation and required no manual tuning of `n_clusters`.

---

## Tools and Libraries

- `sentence-transformers`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `numpy`

---

## Sample Insights

Cluster #18 (from DBSCAN) contained narrative texts with common themes:
- Moral lessons
- Personal struggles
- Adversity and growth

---

## Future Improvements

- Use **HDBSCAN** for better cluster density handling
- Apply **BERTopic** for automatic topic labeling
- Integrate human-based validation of cluster coherence

---

## References

1. Reimers & Gurevych (2019) - Sentence-BERT  
2. PlagBench Dataset (2024)  
3. Scikit-learn Documentation  
4. HuggingFace Transformers & SentenceTransformers  
5. DBSCAN (Ester et al., 1996)

---
