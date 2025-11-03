Step 1: k-NN idea graph (what & why)

Goal: turn today’s raw items (tweets, chats, notes, posts) into a sparse idea graph where nearby nodes mean “semantic neighbors.”

Why: this graph is the substrate for everything else—clustering (breadth), block updates (Gibbs), and later Integration/Entropy metrics.

Intuition: k-NN says “for each idea, keep only its k strongest ties.” That’s how we stay local/sparse and avoid hairball graphs.

How to use it with your own data (easy mode)

Create an items.csv with at least these columns:

id (unique), text (content). Optional: date, type.

Run:

python /mnt/data/knn_pipeline.py --items items.csv --k 5 --min_sim 0.12


This writes knn_nodes.csv and knn_edges.csv (undirected, cosine-similarity weights).

What’s happening inside (plain English)

Embed the text using TF-IDF (bag-of-words + n-grams). It’s simple, transparent, and good enough to start.

Measure similarity via cosine similarity between embeddings.

Keep top-k neighbors per node, symmetrize the edges, and record edge weights.

Result: a small-world-ish graph that captures your day’s conceptual neighborhoods.

Sensible defaults (and when to change them)

k = 5–10: more edges = smoother clusters but more noise. Start at 5.

min_sim = 0.10–0.20: clips weak ties. If your texts are short, drop to ~0.08.

Use per-day graphs for daily IV; also keep a rolling 7-day graph for stability.

Quality checks (fast)

Isolated nodes? (degree 0) → either true one-offs or needs lower min_sim.

One giant component? → lower k or raise min_sim to keep things local.

Duplicate edges/themes? → check for near-duplicates in your items.