# Step 1: k-NN Idea Graph

## What This Step Does

This step transforms raw items (tweets, chats, notes, posts) into a sparse idea graph where nearby nodes represent "semantic neighbors."

**Why it matters:** This graph is the foundation for everything else—clustering, compression, and integration metrics.

**Intuition:** k-NN says "for each idea, keep only its k strongest ties." This creates a sparse, manageable graph that avoids hairball visualizations while preserving conceptual neighborhoods.

## How It Works

1. **Embed text** using TF-IDF (bag-of-words + n-grams)
2. **Measure similarity** via cosine similarity between embeddings
3. **Keep top-k neighbors** per node, symmetrize edges, and record weights
4. **Result:** A small-world graph capturing conceptual neighborhoods

## Input Files

- `items.csv` - Raw items with columns: `id`, `text`, `date`, `type`

## Output Files

### `knn_nodes.csv`
- **What it is:** All nodes (items) in the graph
- **Columns:**
  - `id`: Unique identifier for each item
  - `text`: The actual content (tweet/post text)
  - `date`: Timestamp
  - `type`: Item type (e.g., "tweet")
- **How to interpret:** Each row is an idea node. Nodes connected by edges in `knn_edges.csv` are semantically similar.

### `knn_edges.csv`
- **What it is:** Edges connecting similar ideas
- **Columns:**
  - `source`: ID of first node
  - `target`: ID of second node
  - `similarity`: Cosine similarity weight (0-1, higher = more similar)
- **How to interpret:** Edges represent semantic relationships. Higher similarity = ideas are more conceptually related.

## Parameters

- **k**: Number of nearest neighbors to keep (default: 5)
  - Higher k = more edges = smoother clusters but more noise
  - Start with k=5-10
- **min_sim**: Minimum similarity threshold (default: 0.12)
  - Clips weak ties
  - Lower threshold = more edges, higher threshold = sparser graph

## Quality Checks

- **Isolated nodes (degree 0):** Either true one-offs or needs lower `min_sim`
- **One giant component:** Lower `k` or raise `min_sim` to keep things local
- **Duplicate edges/themes:** Check for near-duplicates in your items

## Usage

```bash
python knn_pipeline.py --items items.csv --k 5 --min_sim 0.12
```

This creates `knn_nodes.csv` and `knn_edges.csv` (undirected, cosine-similarity weights).

## Next Steps

The k-NN graph feeds into:
- **Step 2:** Clustering comparison (community detection + HDBSCAN)
- **Step 4:** Dynamic system modeling (oscillator model)
- **Step 6:** THRML simulation (idea domain formation)

