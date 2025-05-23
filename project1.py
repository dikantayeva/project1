# -*- coding: utf-8 -*-
"""project1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13xu_X8BStRNbpu4DU3ldbZ1nhVJ7FkUt
"""

!pip install -q sentence-transformers

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("susp_docs_cleaned.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned_text"] = df["cleaned_text"].apply(clean_text)

df.head()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df["cleaned_text"].tolist(), show_progress_bar=True)

from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv("susp_docs_cleaned.csv")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(df["cleaned_text"].tolist(), show_progress_bar=True)

pd.DataFrame(embeddings).to_csv("sbert_embeddings.csv", index=False)
print("✅ Готово: sbert_embeddings.csv")

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)


n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(embeddings)

plt.figure(figsize=(10, 7))
for cluster in range(n_clusters):
    subset = emb_2d[df["cluster"] == cluster]
    plt.scatter(subset[:,0], subset[:,1], label=f"Cluster {cluster}")
plt.legend()
plt.title("Clusters (SBERT + t-SNE)")
plt.show()

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

n_clusters = 40

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(embeddings)

plt.figure(figsize=(10, 7))
for cluster in range(n_clusters):
    subset = emb_2d[df["cluster"] == cluster]
    plt.scatter(subset[:,0], subset[:,1], label=f"Cluster {cluster}")
plt.legend()
plt.title("Clusters (SBERT + t-SNE)")
plt.show()

df[["cleaned_text", "cluster"]].to_csv("sbert_clusters.csv", index=False)
files.download("sbert_clusters.csv")

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd

embedding_df = pd.read_csv("sbert_embeddings.csv")
embeddings = embedding_df.values

dbscan = DBSCAN(eps=1.0, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(embeddings)

valid_mask = labels != -1
valid_embeddings = embeddings[valid_mask]
valid_labels = labels[valid_mask]

n_clusters_dbscan = len(set(valid_labels))
sil_score = silhouette_score(valid_embeddings, valid_labels) if n_clusters_dbscan > 1 else None

print(f"Clusters founded: {n_clusters_dbscan}")
print(f"Silhouette Score: {sil_score}")