#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_knn(items_csv: str, edges_csv: str, nodes_csv: str, k: int = 5, min_sim: float = 0.12):
    df = pd.read_csv(items_csv)
    if not {"id","text"}.issubset(df.columns):
        raise ValueError("CSV must contain at least 'id' and 'text' columns.")
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = tfidf.fit_transform(df["text"].astype(str).values)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    neighbors = {}
    for i in range(sim.shape[0]):
        idx = np.argpartition(-sim[i], kth=min(k, sim.shape[0]-1))[:k]
        idx = idx[np.argsort(-sim[i, idx])]
        neighbors[i] = [(int(j), float(sim[i, j])) for j in idx if sim[i, j] >= min_sim]
    edge_dict = {}
    for i, nbrs in neighbors.items():
        for j, s in nbrs:
            a, b = sorted((i, j))
            edge_dict[(a, b)] = max(edge_dict.get((a, b), 0.0), s)
    edges = [{"source": df.loc[a, "id"], "target": df.loc[b, "id"], "similarity": s}
             for (a, b), s in edge_dict.items()]
    pd.DataFrame(edges).to_csv(edges_csv, index=False)
    df.to_csv(nodes_csv, index=False)
    print(f"Wrote {edges_csv} ({len(edges)} edges) and {nodes_csv} ({len(df)} nodes).")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--items", default="items.csv")
    p.add_argument("--edges_out", default="knn_edges.csv")
    p.add_argument("--nodes_out", default="knn_nodes.csv")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min_sim", type=float, default=0.12)
    args = p.parse_args()
    build_knn(args.items, args.edges_out, args.nodes_out, args.k, args.min_sim)
