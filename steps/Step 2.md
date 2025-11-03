Step 2: Clustering Comparison — "Finding Idea Communities"

Goal: Compare two complementary clustering approaches to understand how your ideas cohere from different perspectives.

Why: Each clustering method reveals different aspects of your idea space:
- **Graph-based community detection** → sees emergent themes from relationship patterns
- **HDBSCAN** → sees dense semantic regions in embedding space

Intuition: Different clustering methods tell different stories about the same data.

## 🌐 1. Community Detection (Network Perspective)

**Intuition:**
If two ideas (nodes) are tightly connected by strong similarity edges, they probably belong to the same "conversation."

Community detection partitions the k-NN graph so intra-group edges are dense and inter-group edges are sparse.

**Algorithm:**
- Louvain — fast, classic, works well for 100–10,000 nodes
- Falls back to NetworkX greedy modularity if python-louvain not installed

**Metrics computed:**
- **T_d**: Number of communities
- **Community sizes**: Histogram of cluster sizes
- **H_in_comm**: Normalized entropy = -∑p_i * log(p_i) / log(T_d)

**Interpretation:**
- High entropy → you explored many distinct idea clusters evenly
- Low entropy → focused deeply in a few clusters

## 🔢 2. HDBSCAN (Embedding Perspective)

**Intuition:**
Even if two ideas aren't directly linked, they may sit close in semantic space.

HDBSCAN finds dense blobs of points in the embedding space without needing you to pre-set k.

**Outputs:**
- Cluster label per item (-1 = noise/outlier)
- Cluster probability (confidence score)
- Outlier fraction (novel/experimental ideas)

**Metrics computed:**
- **T_topics**: Number of non-noise clusters
- **H_in_hdb**: Entropy over non-noise clusters
- **Outlier fraction**: Fraction of items marked as noise

**Interpretation:**
- More clusters + moderate outlier rate → diverse exploration
- Mostly one cluster → tight thematic focus
- Many outliers → chaotic exploration (potential new direction)

## ⚖️ 3. Comparing the Two

| Aspect | Community Detection | HDBSCAN |
|--------|---------------------|---------|
| Data basis | Graph edges (similarity network) | Raw embedding coordinates |
| Sensitive to | Connection pattern | Density & shape in space |
| Strength | Captures emergent themes from relationships | Finds fine-grained semantic clusters |
| Weakness | Needs good edge weighting | Sensitive to embedding scaling |
| When to use | To understand idea cohesion | To measure conceptual spread / novelty |

## How to Use

**Prerequisites:**
- `knn_nodes.csv` and `knn_edges.csv` from Step 1
- `items.csv` (original items with text)

**Run:**
```bash
python clustering_comparison.py --nodes knn_nodes.csv --edges knn_edges.csv --items items.csv
```

**Output files:**
- `community_labels.csv` — nodes with community labels
- `hdbscan_labels.csv` — items with HDBSCAN cluster labels and probabilities
- `clustering_metrics.json` — comparison metrics for both methods

**Arguments:**
- `--nodes`: Input nodes CSV (default: `knn_nodes.csv`)
- `--edges`: Input edges CSV (default: `knn_edges.csv`)
- `--items`: Input items CSV (default: `items.csv`)
- `--comm_out`: Output for community labels (default: `community_labels.csv`)
- `--hdbscan_out`: Output for HDBSCAN labels (default: `hdbscan_labels.csv`)
- `--metrics_out`: Output JSON for metrics (default: `clustering_metrics.json`)
- `--min_cluster_size`: HDBSCAN minimum cluster size (default: 2)

## What's Happening Inside

1. **Community Detection:**
   - Loads k-NN graph from nodes and edges
   - Builds NetworkX graph with edge weights
   - Runs Louvain community detection algorithm
   - Computes entropy and community size distribution

2. **HDBSCAN:**
   - Loads items and recreates TF-IDF embeddings (same as Step 1)
   - Optionally applies UMAP dimensionality reduction if features > 50
   - Runs HDBSCAN clustering
   - Computes entropy (excluding noise) and outlier fraction

3. **Comparison:**
   - Compares entropy values
   - Compares cluster counts
   - Provides interpretation based on metrics

## Sensible Defaults

- `min_cluster_size=2`: Minimum size for HDBSCAN clusters (lower = more clusters)
- UMAP reduction: Applied automatically if TF-IDF features > 50
- Both methods use same TF-IDF embeddings for consistency

## Quality Checks

**Community Detection:**
- Too many single-node communities? → Consider lowering k or min_sim in Step 1
- One giant community? → Consider raising min_sim to keep things local

**HDBSCAN:**
- Too many outliers? → Lower min_cluster_size or check for low-quality embeddings
- Only one cluster? → May indicate very coherent theme or need for different embedding

**Comparison:**
- Large entropy difference? → Methods are seeing different structure (expected)
- Very different cluster counts? → Graph structure vs embedding density reveal different patterns

