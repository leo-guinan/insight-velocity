#!/usr/bin/env python3
"""
Two-Pole Adversarial Model for Insight Velocity

Implements the dual-pole architecture:
- Public Pole: Twitter conversations, public interactions
- Private Pole: AI archives, private conversations
- Center: HDBSCAN clustering on both poles
- Adversarial: Domain-agnostic encoder + influence quantification

Based on the design outlined in the system architecture.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML/Stats imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
try:
    import hdbscan
except ImportError:
    hdbscan = None

# Optimal Transport
try:
    from ot import sinkhorn, emd
    from scipy.spatial.distance import cdist
except ImportError:
    # Will degrade gracefully if POT not installed
    sinkhorn = None
    emd = None


class TwoPoleBuilder:
    """
    Builds and aligns two HDBSCAN clusters representing public and private poles.
    """
    
    def __init__(self, min_cluster_size=2, embedding_dim=50):
        self.min_cluster_size = min_cluster_size
        self.embedding_dim = embedding_dim
        self.encoder = None
        
    def build_encoder(self, texts):
        """Build and fit shared TF-IDF encoder for both poles."""
        self.encoder = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2 if len(texts) > 10000 else 1,
            max_features=50000
        )
        # Fit the encoder on the texts
        self.encoder.fit(texts)
        return self.encoder
    
    def encode_texts(self, texts):
        """Encode texts using the shared encoder."""
        if self.encoder is None:
            raise ValueError("Encoder not built. Call build_encoder first.")
        return self.encoder.transform(texts)
    
    def build_pole_graphs(self, public_items, private_items):
        """
        Build k-NN graphs and HDBSCAN clusters for both poles.
        
        Args:
            public_items: List of dicts with 'id', 'text', 'date'
            private_items: List of dicts with 'id', 'text', 'date'
        
        Returns:
            dict with graphs, clusters, and centroids
        """
        print("=" * 80)
        print("🔨 Building Two-Pole Architecture")
        print("=" * 80)
        
        # Combine all texts to build shared encoder
        print(f"\n1️⃣ Building shared encoder...")
        all_texts = [p['text'] for p in public_items + private_items]
        self.build_encoder(all_texts)
        vocab_size = len(self.encoder.vocabulary_) if hasattr(self.encoder, 'vocabulary_') else len(self.encoder.vocabulary)
        print(f"   Encoder vocabulary: {vocab_size} terms")
        
        # Encode both poles
        print(f"\n2️⃣ Encoding poles...")
        public_texts = [p['text'] for p in public_items]
        private_texts = [pr['text'] for pr in private_items]
        
        X_pub = self.encode_texts(public_texts)
        X_priv = self.encode_texts(private_texts)
        
        print(f"   Public: {X_pub.shape[0]} items, {X_pub.shape[1]} features")
        print(f"   Private: {X_priv.shape[0]} items, {X_priv.shape[1]} features")
        
        # Build k-NN graphs
        print(f"\n3️⃣ Building k-NN graphs...")
        G_pub = self._build_knn_graph(public_items, X_pub, k=5, min_sim=0.12)
        G_priv = self._build_knn_graph(private_items, X_priv, k=5, min_sim=0.12)
        
        print(f"   Public graph: {G_pub['n_edges']} edges")
        print(f"   Private graph: {G_priv['n_edges']} edges")
        
        # Build HDBSCAN clusters
        print(f"\n4️⃣ Clustering with HDBSCAN...")
        C_pub = self._build_hdbscan_clusters(public_items, X_pub, 'public')
        C_priv = self._build_hdbscan_clusters(private_items, X_priv, 'private')
        
        # Compute cluster centroids
        print(f"\n5️⃣ Computing cluster centroids...")
        centroids_pub = self._compute_centroids(X_pub, C_pub['labels'])
        centroids_priv = self._compute_centroids(X_priv, C_priv['labels'])
        
        print(f"   Public: {len(centroids_pub)} clusters")
        print(f"   Private: {len(centroids_priv)} clusters")
        
        # Create time series activations
        print(f"\n6️⃣ Creating time series...")
        activations_pub = self._compute_activations(public_items, C_pub['labels'])
        activations_priv = self._compute_activations(private_items, C_priv['labels'])
        
        results = {
            'public': {
                'items': public_items,
                'embeddings': X_pub,
                'graph': G_pub,
                'clusters': C_pub,
                'centroids': centroids_pub,
                'activations': activations_pub
            },
            'private': {
                'items': private_items,
                'embeddings': X_priv,
                'graph': G_priv,
                'clusters': C_priv,
                'centroids': centroids_priv,
                'activations': activations_priv
            }
        }
        
        print(f"\n✅ Two-pole architecture built successfully")
        print("=" * 80)
        
        return results
    
    def _build_knn_graph(self, items, X, k=5, min_sim=0.12):
        """Build k-NN graph from embeddings."""
        # Compute pairwise similarities
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0.0)
        
        # Find k-nearest neighbors
        n_items = sim.shape[0]
        edges = []
        
        for i in range(n_items):
            # Get top k neighbors
            idx = np.argpartition(-sim[i], kth=min(k, n_items-1))[:min(k+1, n_items)]
            idx = idx[np.argsort(-sim[i, idx])]
            
            # Filter by min similarity
            for j in idx:
                if i != j and sim[i, j] >= min_sim:
                    edges.append({
                        'source': items[i]['id'],
                        'target': items[j]['id'],
                        'similarity': float(sim[i, j])
                    })
        
        # Deduplicate bidirectional edges
        edge_dict = {}
        for edge in edges:
            key = tuple(sorted([edge['source'], edge['target']]))
            if key not in edge_dict or edge['similarity'] > edge_dict[key]['similarity']:
                edge_dict[key] = edge
        
        return {
            'edges': list(edge_dict.values()),
            'n_edges': len(edge_dict),
            'n_nodes': n_items
        }
    
    def _build_hdbscan_clusters(self, items, X, pole_name):
        """Build HDBSCAN clusters on embeddings."""
        if hdbscan is None:
            raise ImportError("HDBSCAN not installed. Install with: pip install hdbscan")
        
        # Reduce dimensionality if needed
        if X.shape[1] > 100:
            # Use TruncatedSVD for sparse-friendly reduction
            try:
                from sklearn.decomposition import TruncatedSVD
                if X.shape[1] > 1000:
                    print(f"   Reducing {X.shape[1]} → 100 dims...")
                    svd = TruncatedSVD(n_components=100, random_state=42)
                    X_red = svd.fit_transform(X)
                else:
                    X_red = X.toarray() if hasattr(X, 'toarray') else X
                
                # Then UMAP if available
                try:
                    import umap
                    print(f"   Reducing to {self.embedding_dim} dims with UMAP...")
                    reducer = umap.UMAP(n_components=self.embedding_dim, random_state=42, verbose=False)
                    X_final = reducer.fit_transform(X_red)
                except ImportError:
                    # Further SVD reduction
                    if X_red.shape[1] > self.embedding_dim:
                        svd2 = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
                        X_final = svd2.fit_transform(X_red)
                    else:
                        X_final = X_red
            except Exception as e:
                print(f"   Warning: Dimensionality reduction failed ({e}), using raw")
                X_final = X.toarray() if hasattr(X, 'toarray') else X
        else:
            X_final = X.toarray() if hasattr(X, 'toarray') else X
        
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1
        )
        labels = clusterer.fit_predict(X_final)
        probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
        
        # Count clusters
        cluster_counts = Counter(labels)
        noise_count = cluster_counts.get(-1, 0)
        non_noise_clusters = {k: v for k, v in cluster_counts.items() if k != -1}
        
        print(f"   {pole_name}: {len(non_noise_clusters)} clusters, {noise_count} outliers")
        
        return {
            'labels': labels,
            'probabilities': probabilities,
            'cluster_counts': dict(cluster_counts),
            'n_clusters': len(non_noise_clusters),
            'n_outliers': noise_count
        }
    
    def _compute_centroids(self, X, labels):
        """Compute centroids for each cluster."""
        centroids = {}
        unique_labels = set(labels)
        
        # Convert sparse to dense if needed
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            mask = labels == label
            if mask.sum() > 0:
                centroids[label] = X_dense[mask].mean(axis=0)
        
        return centroids
    
    def _compute_activations(self, items, labels):
        """Compute daily activation time series for each cluster."""
        # Parse dates and create time series
        dates = {}
        for item, label in zip(items, labels):
            if label == -1:  # Skip outliers
                continue
            date = item.get('date')
            if date:
                dates[date] = dates.get(date, Counter())
                dates[date][label] += 1
        
        # Convert to DataFrame
        if not dates:
            return pd.DataFrame()
        
        activations = pd.DataFrame(dates).T
        activations.index.name = 'date'
        activations = activations.fillna(0).sort_index()
        
        return activations


class ClusterAligner:
    """
    Aligns clusters across two poles using optimal transport.
    """
    
    def __init__(self, reg=1.0, use_emd=True):
        self.reg = reg  # Regularization for Sinkhorn
        self.use_emd = use_emd  # Use EMD vs Sinkhorn
        self.alignment_map = None
    
    def align_clusters(self, centroids_pub, centroids_priv, activations_pub, activations_priv):
        """
        Align clusters using optimal transport.
        
        Args:
            centroids_pub: dict mapping cluster_id -> embedding vector
            centroids_priv: dict mapping cluster_id -> embedding vector
            activations_pub: DataFrame with cluster activations over time
            activations_priv: DataFrame with cluster activations over time
        
        Returns:
            dict with alignment map and metrics
        """
        print("\n" + "=" * 80)
        print("🔗 Cluster Alignment via Optimal Transport")
        print("=" * 80)
        
        if not centroids_pub or not centroids_priv:
            print("   No clusters to align")
            return None
        
        # Build similarity matrix
        print(f"\n1️⃣ Computing cross-pole similarity matrix...")
        pub_labels = sorted(centroids_pub.keys())
        priv_labels = sorted(centroids_priv.keys())
        
        S = np.zeros((len(pub_labels), len(priv_labels)))
        for i, k in enumerate(pub_labels):
            for j, m in enumerate(priv_labels):
                sim = cosine_similarity(
                    [centroids_pub[k]], 
                    [centroids_priv[m]]
                )[0, 0]
                S[i, j] = sim
        
        print(f"   Similarity matrix: {S.shape[0]} × {S.shape[1]}")
        print(f"   Max similarity: {S.max():.4f}, Mean: {S.mean():.4f}")
        
        # Compute marginal distributions (weighted by activations)
        print(f"\n2️⃣ Computing marginal distributions...")
        p = self._compute_marginal(activations_pub, pub_labels)
        q = self._compute_marginal(activations_priv, priv_labels)
        
        print(f"   Public marginals: {len(p)} clusters, sum={p.sum():.4f}")
        print(f"   Private marginals: {len(q)} clusters, sum={q.sum():.4f}")
        
        # Solve optimal transport
        print(f"\n3️⃣ Solving optimal transport...")
        if emd is not None and self.use_emd:
            # Use EMD (exact)
            M = 1.0 - S  # Cost matrix: 1 - similarity
            pi_star = emd(p, q, M, numItermax=1000000)
        elif sinkhorn is not None:
            # Use Sinkhorn (approximate, fast)
            M = 1.0 - S
            pi_star = sinkhorn(p, q, M, self.reg, numItermax=1000)
        else:
            print("   Warning: Optimal transport library not found. Using greedy alignment.")
            pi_star = self._greedy_alignment(S, p, q)
        
        self.alignment_map = pi_star
        
        # Compute alignment metrics
        alignment_strength = np.sum(pi_star * S)
        total_mass = pi_star.sum()
        
        # Find best alignments
        # Diagnostic: check mass distribution
        all_masses = pi_star[pi_star > 0].flatten()
        if len(all_masses) > 0:
            print(f"   Mass distribution: max={all_masses.max():.6f}, "
                  f"mean={all_masses.mean():.6f}, median={np.median(all_masses):.6f}")
            print(f"   Non-zero pairs: {len(all_masses)} / {pi_star.size}")
        
        # Use adaptive threshold: if max mass is small, use a percentage of max
        # Otherwise use 0.01 as baseline
        if len(all_masses) > 0:
            max_mass = all_masses.max()
            if max_mass < 0.01:
                # When masses are very small, use top 1% of non-zero masses
                # or 50% of max, whichever is higher
                percentile_threshold = np.percentile(all_masses, 99)
                mass_threshold = max(percentile_threshold, max_mass * 0.5)
                print(f"   ⚠️  Small masses detected (max={max_mass:.6f}), using adaptive threshold")
            else:
                mass_threshold = 0.01
        else:
            mass_threshold = 0.01
        
        print(f"   Using mass threshold: {mass_threshold:.6f}")
        
        alignments = []
        for i, k in enumerate(pub_labels):
            for j, m in enumerate(priv_labels):
                mass = pi_star[i, j]
                if mass > mass_threshold:
                    alignments.append({
                        'public_cluster': k,
                        'private_cluster': m,
                        'mass': float(mass),
                        'similarity': float(S[i, j])
                    })
        
        alignments = sorted(alignments, key=lambda x: x['mass'], reverse=True)
        
        print(f"\n   Alignment strength: {alignment_strength:.4f}")
        print(f"   Total matched mass: {total_mass:.4f}")
        print(f"   Alignments (mass > {mass_threshold:.6f}): {len(alignments)}")
        print(f"   Strong alignments (mass > 0.1): {len([a for a in alignments if a['mass'] > 0.1])}")
        
        # If no alignments found, suggest diagnostics
        if len(alignments) == 0 and len(all_masses) > 0:
            print(f"\n   ⚠️  No alignments found with threshold {mass_threshold:.6f}")
            print(f"   Top 10 masses: {sorted(all_masses, reverse=True)[:10]}")
            print(f"   Consider:")
            print(f"     - Increasing min_cluster_size (fewer, larger clusters)")
            print(f"     - Adjusting date range (more overlap in time)")
            print(f"     - Using time-aware alignment (combines semantic + temporal)")
        
        # Identify unmapped clusters
        unmapped_pub = []
        unmapped_priv = []
        
        pub_mapped_mass = pi_star.sum(axis=1)
        for i, k in enumerate(pub_labels):
            if pub_mapped_mass[i] < 0.1:
                unmapped_pub.append(k)
        
        priv_mapped_mass = pi_star.sum(axis=0)
        for j, m in enumerate(priv_labels):
            if priv_mapped_mass[j] < 0.1:
                unmapped_priv.append(m)
        
        print(f"\n   Unmapped public: {len(unmapped_pub)} clusters")
        print(f"   Unmapped private: {len(unmapped_priv)} clusters")
        
        results = {
            'alignment_map': pi_star,
            'alignments': alignments,
            'alignment_strength': float(alignment_strength),
            'unmapped_public': unmapped_pub,
            'unmapped_private': unmapped_priv,
            'similarity_matrix': S,
            'marginals': {'public': p, 'private': q}
        }
        
        print("\n✅ Cluster alignment complete")
        print("=" * 80)
        
        return results
    
    def _compute_marginal(self, activations, labels):
        """Compute marginal distribution over clusters from activations."""
        if activations.empty:
            # Uniform if no data
            return np.ones(len(labels)) / len(labels)
        
        # Sum activations across time
        total_activation = activations.sum().sum()
        if total_activation == 0:
            return np.ones(len(labels)) / len(labels)
        
        # Weight by activation
        marginals = np.array([
            activations.get(label, pd.Series()).sum() if label in activations.columns else 0
            for label in labels
        ])
        
        # Normalize
        if marginals.sum() > 0:
            marginals = marginals / marginals.sum()
        else:
            marginals = np.ones(len(labels)) / len(labels)
        
        return marginals
    
    def _greedy_alignment(self, S, p, q):
        """Greedy alignment when OT library is unavailable."""
        # Sort by similarity and greedily assign
        n_pub, n_priv = S.shape
        pi = np.zeros((n_pub, n_priv))
        
        # Greedy matching
        pairs = []
        for i in range(n_pub):
            for j in range(n_priv):
                pairs.append((i, j, S[i, j] * p[i] * q[j]))
        
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        
        # Assign greedily
        pub_assigned = set()
        priv_assigned = set()
        remaining_p, remaining_q = p.copy(), q.copy()
        
        for i, j, _ in pairs:
            if i not in pub_assigned and j not in priv_assigned:
                # Assign mass based on marginals
                mass = min(remaining_p[i], remaining_q[j])
                pi[i, j] = mass
                remaining_p[i] -= mass
                remaining_q[j] -= mass
                
                if remaining_p[i] < 1e-6:
                    pub_assigned.add(i)
                if remaining_q[j] < 1e-6:
                    priv_assigned.add(j)
        
        return pi


class DirectionalInfluence:
    """
    Measures directional influence between poles using lagged correlation
    and transfer entropy.
    """
    
    def compute_influence_metrics(self, alignments, activations_pub, activations_priv, max_lag=14):
        """
        Compute directional influence for each aligned pair.
        
        Args:
            alignments: List of dicts from ClusterAligner
            activations_pub: DataFrame with public activations
            activations_priv: DataFrame with private activations
            max_lag: Maximum lag in days to test
        
        Returns:
            DataFrame with influence metrics per alignment
        """
        print("\n" + "=" * 80)
        print("📊 Computing Directional Influence")
        print("=" * 80)
        
        results = []
        
        for alignment in alignments:
            k = alignment['public_cluster']
            m = alignment['private_cluster']
            
            # Skip if we don't have enough data
            if not self._has_activation(k, activations_pub) or \
               not self._has_activation(m, activations_priv):
                continue
            
            # Compute lagged correlation
            rho = self._compute_lagged_correlation(
                activations_priv, m,
                activations_pub, k,
                max_lag=max_lag
            )
            
            # Compute transfer entropy
            te_priv_to_pub = self._compute_transfer_entropy(
                activations_priv, m,
                activations_pub, k
            )
            te_pub_to_priv = self._compute_transfer_entropy(
                activations_pub, k,
                activations_priv, m
            )
            
            results.append({
                'public_cluster': k,
                'private_cluster': m,
                'alignment_mass': alignment['mass'],
                'similarity': alignment['similarity'],
                'lag': rho['optimal_lag'],
                'correlation_at_lag': rho['correlation'],
                'te_priv_to_pub': te_priv_to_pub,
                'te_pub_to_priv': te_pub_to_priv,
                'net_influence': te_priv_to_pub - te_pub_to_priv
            })
        
        if not results:
            print("   No alignments with sufficient data")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('net_influence', ascending=False)
        
        print(f"\n   Analyzed {len(df)} alignments")
        print(f"   Strong influence (|net| > 0.1): {(df['net_influence'].abs() > 0.1).sum()}")
        print(f"   Private → Public dominant: {(df['net_influence'] > 0).sum()}")
        print(f"   Public → Private dominant: {(df['net_influence'] < 0).sum()}")
        
        print("\n✅ Influence metrics complete")
        print("=" * 80)
        
        return df
    
    def _has_activation(self, cluster_id, activations):
        """Check if cluster has sufficient activation data."""
        if cluster_id in activations.columns:
            return activations[cluster_id].sum() > 0
        return False
    
    def _compute_lagged_correlation(self, X, x_col, Y, y_col, max_lag=14):
        """
        Compute lagged correlation to determine lead/lag relationship.
        Returns optimal lag and correlation.
        """
        x_series = X[x_col] if x_col in X.columns else pd.Series()
        y_series = Y[y_col] if y_col in Y.columns else pd.Series()
        
        # Align indices (dates)
        common_dates = x_series.index.intersection(y_series.index)
        if len(common_dates) < 10:
            return {'optimal_lag': 0, 'correlation': 0.0}
        
        x_aligned = x_series[common_dates]
        y_aligned = y_series[common_dates]
        
        best_corr = -np.inf
        best_lag = 0
        
        # Test various lags
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                x_shifted = x_aligned
                y_shifted = y_aligned
            elif lag > 0:
                # X leads Y by lag days
                x_shifted = x_aligned[:-lag] if lag < len(x_aligned) else pd.Series()
                y_shifted = y_aligned[lag:]
            else:
                # Y leads X by |lag| days
                x_shifted = x_aligned[-lag:]
                y_shifted = y_aligned[:lag] if lag < len(y_aligned) else pd.Series()
            
            if len(x_shifted) < 5 or len(y_shifted) < 5:
                continue
            
            # Ensure same length
            min_len = min(len(x_shifted), len(y_shifted))
            x_shifted = x_shifted[:min_len]
            y_shifted = y_shifted[:min_len]
            
            try:
                corr, p_val = pearsonr(x_shifted, y_shifted)
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            except:
                pass
        
        return {'optimal_lag': best_lag, 'correlation': float(best_corr)}
    
    def _compute_transfer_entropy(self, X, x_col, Y, y_col, lag=1):
        """
        Approximate transfer entropy using correlation.
        Full TE would require mutual information, this is a proxy.
        """
        x_series = X[x_col] if x_col in X.columns else pd.Series()
        y_series = Y[y_col] if y_col in Y.columns else pd.Series()
        
        common_dates = x_series.index.intersection(y_series.index)
        if len(common_dates) < 10:
            return 0.0
        
        x_aligned = x_series[common_dates]
        y_aligned = y_series[common_dates]
        
        # Simple proxy: squared correlation
        try:
            corr, _ = pearsonr(x_aligned, y_aligned)
            return float(corr ** 2) if not np.isnan(corr) else 0.0
        except:
            return 0.0


def main():
    """CLI for two-pole modeling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Two-Pole Adversarial Model: Public vs Private Influence Analysis'
    )
    parser.add_argument('--public', required=True, help='CSV with public items (id, text, date)')
    parser.add_argument('--private', required=True, help='CSV with private items (id, text, date)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--min_cluster_size', type=int, default=2, help='HDBSCAN min_cluster_size')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    public_df = pd.read_csv(args.public)
    private_df = pd.read_csv(args.private)
    
    public_items = public_df.to_dict('records')
    private_items = private_df.to_dict('records')
    
    # Build two-pole architecture
    builder = TwoPoleBuilder(min_cluster_size=args.min_cluster_size)
    results = builder.build_pole_graphs(public_items, private_items)
    
    # Align clusters
    aligner = ClusterAligner()
    alignment = aligner.align_clusters(
        results['public']['centroids'],
        results['private']['centroids'],
        results['public']['activations'],
        results['private']['activations']
    )
    
    # Compute influence
    if alignment:
        influence_calculator = DirectionalInfluence()
        influence_df = influence_calculator.compute_influence_metrics(
            alignment['alignments'],
            results['public']['activations'],
            results['private']['activations']
        )
    else:
        influence_df = pd.DataFrame()
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving outputs to {output_dir}...")
    
    # Save cluster labels
    pd.DataFrame({
        'id': [i['id'] for i in public_items],
        'label': results['public']['clusters']['labels']
    }).to_csv(output_dir / 'public_clusters.csv', index=False)
    
    pd.DataFrame({
        'id': [i['id'] for i in private_items],
        'label': results['private']['clusters']['labels']
    }).to_csv(output_dir / 'private_clusters.csv', index=False)
    
    # Save alignment results
    if influence_df is not None and not influence_df.empty:
        influence_df.to_csv(output_dir / 'influence_map.csv', index=False)
    
    # Save time series
    results['public']['activations'].to_csv(output_dir / 'public_activations.csv')
    results['private']['activations'].to_csv(output_dir / 'private_activations.csv')
    
    # Save summary
    summary = {
        'public': {
            'n_items': len(public_items),
            'n_clusters': results['public']['clusters']['n_clusters'],
            'n_outliers': results['public']['clusters']['n_outliers']
        },
        'private': {
            'n_items': len(private_items),
            'n_clusters': results['private']['clusters']['n_clusters'],
            'n_outliers': results['private']['clusters']['n_outliers']
        },
        'alignment': {
            'alignment_strength': alignment['alignment_strength'] if alignment else None,
            'n_alignments': len(alignment['alignments']) if alignment else 0,
            'unmapped_public': len(alignment['unmapped_public']) if alignment else 0,
            'unmapped_private': len(alignment['unmapped_private']) if alignment else 0
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Complete! Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

