# Step 2: Clustering Comparison

## What This Step Does

Compares two complementary clustering approaches to understand how ideas cohere from different perspectives:

1. **Graph-based community detection (Louvain)** → Sees emergent themes from relationship patterns
2. **HDBSCAN clustering** → Finds dense semantic regions in embedding space

**Why it matters:** Each method reveals different aspects of your idea space. Together they provide a more complete picture of how ideas cluster and cohere.

## How It Works

### Community Detection (Network Perspective)
- **Algorithm:** Louvain community detection on the k-NN graph
- **Input:** Graph edges with similarity weights
- **Output:** Communities based on network structure
- **Intuition:** If two ideas are tightly connected, they belong to the same "conversation"

### HDBSCAN (Embedding Perspective)
- **Algorithm:** HDBSCAN density-based clustering
- **Input:** TF-IDF embeddings from items
- **Output:** Clusters based on semantic space density
- **Intuition:** Ideas close in embedding space cluster together, even without direct graph links

## Input Files

- `knn_nodes.csv` - Nodes from Step 1
- `knn_edges.csv` - Edges from Step 1
- `items.csv` - Original items with text

## Output Files

### `community_labels.csv`
- **What it is:** Nodes with community detection labels
- **Columns:**
  - All columns from `knn_nodes.csv`
  - `community_label`: Integer community ID (0, 1, 2, ...)
- **How to interpret:** Items with the same `community_label` are part of the same community. Higher community count = more diverse exploration.

### `hdbscan_labels.csv`
- **What it is:** Items with HDBSCAN cluster labels
- **Columns:**
  - All columns from `items.csv`
  - `hdbscan_label`: Integer cluster ID (-1 = noise/outlier)
  - `hdbscan_probability`: Confidence score (0-1)
- **How to interpret:** 
  - Non-negative labels = clusters (higher count = more diverse)
  - Label -1 = outliers (novel/experimental ideas)
  - Higher outlier fraction = more exploration/novelty

### `clustering_metrics.json`
- **What it is:** Comparison metrics for both methods
- **Metrics:**
  - `community_detection.T_d`: Number of communities
  - `community_detection.H_in_comm`: Normalized entropy (0-1, higher = more diverse)
  - `hdbscan.T_topics`: Number of non-noise clusters
  - `hdbscan.H_in_hdb`: Entropy over non-noise clusters
  - `hdbscan.outlier_fraction`: Fraction of outliers (0-1)
- **How to interpret:**
  - High entropy (both methods) = diverse exploration
  - Low entropy = focused clusters
  - High outlier fraction (HDBSCAN) = many novel ideas

## Key Metrics Explained

### Entropy (H_in_comm / H_in_hdb)
- **Formula:** `H = -∑p_i * log(p_i) / log(T)` where T = number of clusters
- **Range:** 0-1 (normalized)
- **Interpretation:**
  - High entropy (>0.7): Explored many distinct idea clusters evenly
  - Low entropy (<0.5): Focused deeply in a few clusters
  - Near 1.0: Perfectly balanced exploration

### Outlier Fraction
- **Formula:** `outliers / total_items`
- **Range:** 0-1
- **Interpretation:**
  - Low (<0.1): Mostly coherent clusters
  - Moderate (0.1-0.3): Some novel/experimental ideas
  - High (>0.3): Chaotic exploration or potential new directions

## Comparison Table

| Aspect | Community Detection | HDBSCAN |
|--------|---------------------|---------|
| **Data basis** | Graph edges (similarity network) | Raw embedding coordinates |
| **Sensitive to** | Connection pattern | Density & shape in space |
| **Strength** | Captures emergent themes from relationships | Finds fine-grained semantic clusters |
| **Weakness** | Needs good edge weighting | Sensitive to embedding scaling |
| **When to use** | To understand idea cohesion | To measure conceptual spread / novelty |

## Usage

```bash
python clustering_comparison.py --nodes knn_nodes.csv --edges knn_edges.csv --items items.csv
```

## Next Steps

Clustering results feed into:
- **Step 3:** IV metrics calculation (uses entropy and outlier fraction)
- **Step 4:** Dynamic system modeling (uses cluster counts)
- **Step 6:** THRML simulation (uses clusters as initial states)

