#!/usr/bin/env python3
"""
Advanced Temporal Analysis for Two-Pole Model

Implements:
1. Lead-lag map per cluster pair (cross-correlation, transfer entropy, DII)
2. Per-topic learning lag and inverse learning curve (ILC)
3. Survival/hazard modeling (time-to-externalization)
4. Hawkes point process for causal lag estimation
5. Time-aware alignment (semantic + temporal)
6. Visualization utilities
7. Time-aware blind spot detection
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class AdvancedTemporalAnalyzer:
    """
    Advanced temporal analysis for two-pole model.
    """
    
    def __init__(self, similarity_threshold=0.25, max_lag=14, rho_min=0.2):
        self.similarity_threshold = similarity_threshold
        self.max_lag = max_lag
        self.rho_min = rho_min
    
    def compute_lead_lag_map(self, centroids_pub, centroids_priv, 
                            activations_pub, activations_priv, 
                            similarity_matrix=None):
        """
        Build lead-lag map for each cluster pair with cosine similarity >= threshold.
        
        Returns DataFrame with columns:
        - private_cluster, public_cluster
        - cosine_similarity
        - optimal_lag (tau*)
        - correlation_at_lag (rho)
        - te_priv_to_pub
        - te_pub_to_priv
        - dii (Directional Influence Index)
        """
        print("\n" + "=" * 80)
        print("📊 Computing Lead-Lag Map")
        print("=" * 80)
        
        if similarity_matrix is None:
            # Compute cosine similarity matrix
            print("   Computing similarity matrix...")
            similarity_matrix = self._compute_similarity_matrix(
                centroids_pub, centroids_priv
            )
        
        pub_labels = sorted(centroids_pub.keys())
        priv_labels = sorted(centroids_priv.keys())
        
        results = []
        
        for i, k in enumerate(pub_labels):
            for j, m in enumerate(priv_labels):
                cosine = similarity_matrix[i, j]
                
                # Only process pairs above threshold
                if cosine < self.similarity_threshold:
                    continue
                
                # Get activation series
                a_priv = self._get_activation_series(activations_priv, m)
                a_pub = self._get_activation_series(activations_pub, k)
                
                if len(a_priv) < 10 or len(a_pub) < 10:
                    continue
                
                # Cross-correlation over lag window
                rho_result = self._cross_correlation(a_priv, a_pub, self.max_lag)
                tau_star = rho_result['optimal_lag']
                rho_at_tau = rho_result['correlation']
                
                # Only keep if correlation meets threshold
                if abs(rho_at_tau) < self.rho_min:
                    continue
                
                # Transfer entropy
                te_priv_to_pub = self._compute_transfer_entropy(a_priv, a_pub)
                te_pub_to_priv = self._compute_transfer_entropy(a_pub, a_priv)
                
                # Directional Influence Index
                dii = te_priv_to_pub - te_pub_to_priv
                
                results.append({
                    'private_cluster': m,
                    'public_cluster': k,
                    'cosine_similarity': float(cosine),
                    'optimal_lag': int(tau_star),
                    'correlation_at_lag': float(rho_at_tau),
                    'te_priv_to_pub': float(te_priv_to_pub),
                    'te_pub_to_priv': float(te_pub_to_priv),
                    'dii': float(dii),
                    'private_leads': tau_star > 0
                })
        
        if not results:
            print("   No valid pairs found above thresholds")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('dii', ascending=False)
        
        print(f"\n   Analyzed {len(df)} cluster pairs")
        print(f"   Private → Public (tau* > 0): {(df['optimal_lag'] > 0).sum()}")
        print(f"   Public → Private (tau* < 0): {(df['optimal_lag'] < 0).sum()}")
        print(f"   Strong influence (|DII| > 0.1): {(df['dii'].abs() > 0.1).sum()}")
        
        return df
    
    def compute_per_topic_learning_lag(self, lead_lag_map):
        """
        Compute per-topic learning lag and inverse learning curve (ILC).
        
        For each private cluster m:
        - Weighted lag: median of tau*_mk weighted by rho * cosine
        - ILC_m = 1 / (L_m + 1)
        - Penalize instability with variance term
        """
        print("\n" + "=" * 80)
        print("📈 Computing Per-Topic Learning Lag")
        print("=" * 80)
        
        if lead_lag_map.empty:
            return pd.DataFrame()
        
        # Group by private cluster
        topic_metrics = []
        
        for m in lead_lag_map['private_cluster'].unique():
            subset = lead_lag_map[lead_lag_map['private_cluster'] == m]
            
            # Weights: correlation * cosine similarity
            weights = subset['correlation_at_lag'].abs() * subset['cosine_similarity']
            lags = subset['optimal_lag'].values
            
            # Weighted median
            if len(lags) > 0:
                sorted_indices = np.argsort(lags)
                sorted_lags = lags[sorted_indices]
                sorted_weights = weights.values[sorted_indices]
                cumsum_weights = np.cumsum(sorted_weights)
                total_weight = cumsum_weights[-1]
                
                median_idx = np.searchsorted(cumsum_weights, total_weight / 2)
                L_m = sorted_lags[median_idx] if median_idx < len(sorted_lags) else sorted_lags[-1]
                
                # Variance of lags (instability measure)
                sigma_tau = np.std(lags)
                
                # Inverse Learning Curve
                ILC_m = 1.0 / (L_m + 1.0)
                
                # Penalize instability
                ILC_m_penalized = ILC_m * np.exp(-sigma_tau)
                
                topic_metrics.append({
                    'private_cluster': m,
                    'weighted_median_lag': float(L_m),
                    'lag_variance': float(sigma_tau),
                    'ilc': float(ILC_m),
                    'ilc_penalized': float(ILC_m_penalized),
                    'n_aligned_publics': len(subset)
                })
        
        if not topic_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(topic_metrics)
        df = df.sort_values('ilc_penalized', ascending=False)
        
        print(f"\n   Computed ILC for {len(df)} private topics")
        print(f"   Mean learning lag: {df['weighted_median_lag'].mean():.2f} days")
        print(f"   Mean ILC (penalized): {df['ilc_penalized'].mean():.4f}")
        
        return df
    
    def survival_analysis(self, activations_priv, activations_pub, 
                         lead_lag_map, cluster_metadata=None):
        """
        Survival/hazard modeling for time-to-externalization.
        
        Uses Kaplan-Meier and Cox proportional hazards.
        """
        print("\n" + "=" * 80)
        print("⏱️  Survival Analysis (Time-to-Externalization)")
        print("=" * 80)
        
        if not LIFELINES_AVAILABLE:
            print("   ⚠️  lifelines not installed. Skipping survival analysis.")
            print("   Install with: pip install lifelines")
            return None
        
        if lead_lag_map.empty:
            return None
        
        # Build survival data
        survival_data = []
        
        for m in activations_priv.columns:
            if m == 'date':
                continue
            
            # Find aligned public clusters
            aligned = lead_lag_map[lead_lag_map['private_cluster'] == m]
            if aligned.empty:
                continue
            
            # Get private activation times
            priv_activations = activations_priv[m]
            priv_active_days = priv_activations[priv_activations > 0].index.tolist()
            
            if not priv_active_days:
                continue
            
            # Risk starts at first private activation
            risk_start = pd.to_datetime(min(priv_active_days))
            
            # Check for public activation in aligned clusters
            public_clusters = aligned['public_cluster'].unique()
            public_active_days = []
            
            for k in public_clusters:
                if k in activations_pub.columns:
                    pub_activations = activations_pub[k]
                    public_active_days.extend(
                        pub_activations[pub_activations > 0].index.tolist()
                    )
            
            if public_active_days:
                event_time = pd.to_datetime(min(public_active_days))
                time_to_event = (event_time - risk_start).days
                event_occurred = 1
            else:
                # Censored (no public activation yet)
                time_to_event = (pd.to_datetime(max(priv_active_days)) - risk_start).days
                event_occurred = 0
            
            # Get metadata for covariates
            row = {
                'private_cluster': m,
                'time_to_event': time_to_event,
                'event_occurred': event_occurred
            }
            
            # Add covariates if available
            if cluster_metadata:
                if m in cluster_metadata:
                    row.update(cluster_metadata[m])
            
            survival_data.append(row)
        
        if not survival_data:
            print("   No survival data generated")
            return None
        
        df_survival = pd.DataFrame(survival_data)
        
        # Kaplan-Meier estimation
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=df_survival['time_to_event'],
            event_observed=df_survival['event_occurred']
        )
        
        median_survival = kmf.median_survival_time_
        print(f"\n   Median time-to-externalization: {median_survival:.1f} days")
        
        # Cox model if we have covariates
        if len(df_survival.columns) > 3:
            try:
                cox = CoxPHFitter()
                # Use numeric columns as covariates
                numeric_cols = df_survival.select_dtypes(include=[np.number]).columns
                covariate_cols = [c for c in numeric_cols if c not in ['time_to_event', 'event_occurred']]
                
                if covariate_cols:
                    cox.fit(
                        df_survival,
                        duration_col='time_to_event',
                        event_col='event_occurred'
                    )
                    print(f"\n   Cox Model Summary:")
                    print(cox.summary)
            except Exception as e:
                print(f"   ⚠️  Cox model fitting failed: {e}")
        
        return {
            'survival_data': df_survival,
            'kmf': kmf,
            'median_survival': median_survival
        }
    
    def hawkes_point_process(self, activations_priv, activations_pub,
                            lead_lag_map):
        """
        Fit Hawkes point process to estimate causal lag and strength.
        
        Model: λ_pub(t) = μ + α * Σ κ(t - t_i) where κ(Δ) = β * exp(-βΔ)
        
        Returns per-cluster-pair estimates of (α, β).
        """
        print("\n" + "=" * 80)
        print("⚡ Hawkes Point Process Analysis")
        print("=" * 80)
        
        if lead_lag_map.empty:
            return pd.DataFrame()
        
        results = []
        
        for _, row in lead_lag_map.iterrows():
            m = row['private_cluster']
            k = row['public_cluster']
            
            # Get event times
            priv_events = self._get_event_times(activations_priv, m)
            pub_events = self._get_event_times(activations_pub, k)
            
            if len(priv_events) < 3 or len(pub_events) < 2:
                continue
            
            # Fit Hawkes process
            try:
                alpha, beta = self._fit_hawkes(priv_events, pub_events)
                
                # Characteristic lag = 1/β
                char_lag = 1.0 / beta if beta > 0 else np.inf
                
                results.append({
                    'private_cluster': m,
                    'public_cluster': k,
                    'hawkes_alpha': float(alpha),
                    'hawkes_beta': float(beta),
                    'characteristic_lag': float(char_lag),
                    'causal_strength': float(alpha),  # α > 0 means private causes public
                    'learning_speed': float(beta)  # Higher β = faster learning
                })
            except Exception as e:
                continue
        
        if not results:
            print("   No valid Hawkes fits")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        print(f"\n   Fitted Hawkes process for {len(df)} pairs")
        print(f"   Mean α (causal strength): {df['hawkes_alpha'].mean():.4f}")
        print(f"   Mean β (learning speed): {df['hawkes_beta'].mean():.4f}")
        print(f"   Mean characteristic lag: {df['characteristic_lag'].mean():.2f} days")
        
        return df
    
    def time_aware_alignment(self, centroids_pub, centroids_priv,
                            activations_pub, activations_priv,
                            gamma=0.5):
        """
        Time-aware alignment combining semantic and temporal similarity.
        
        Combined score: J_mk = S_mk^γ * C_mk^(1-γ)
        where C_mk = max_τ ρ_mk(τ)
        """
        print("\n" + "=" * 80)
        print("🔗 Time-Aware Alignment")
        print("=" * 80)
        
        # Compute semantic similarity
        S = self._compute_similarity_matrix(centroids_pub, centroids_priv)
        
        # Compute temporal coupling (max cross-correlation)
        print("   Computing temporal coupling...")
        pub_labels = sorted(centroids_pub.keys())
        priv_labels = sorted(centroids_priv.keys())
        C = np.zeros_like(S)
        
        for i, k in enumerate(pub_labels):
            for j, m in enumerate(priv_labels):
                if S[i, j] < self.similarity_threshold:
                    C[i, j] = 0.0
                    continue
                
                a_priv = self._get_activation_series(activations_priv, m)
                a_pub = self._get_activation_series(activations_pub, k)
                
                if len(a_priv) < 10 or len(a_pub) < 10:
                    C[i, j] = 0.0
                    continue
                
                rho_result = self._cross_correlation(a_priv, a_pub, self.max_lag)
                C[i, j] = abs(rho_result['max_correlation'])
        
        # Combined score
        # Handle zeros (avoid log(0))
        S_safe = np.clip(S, 1e-10, 1.0)
        C_safe = np.clip(C, 1e-10, 1.0)
        
        J = (S_safe ** gamma) * (C_safe ** (1 - gamma))
        
        print(f"   Combined scores computed (γ={gamma})")
        print(f"   Mean semantic similarity: {S[S > 0].mean():.4f}")
        print(f"   Mean temporal coupling: {C[C > 0].mean():.4f}")
        print(f"   Mean combined score: {J[J > 0].mean():.4f}")
        
        return {
            'combined_matrix': J,
            'semantic_matrix': S,
            'temporal_matrix': C,
            'gamma': gamma
        }
    
    def detect_time_aware_blind_spots(self, activations_priv, activations_pub,
                                     lead_lag_map, topic_ilc, 
                                     synthesis_predictions=None,
                                     recency_window_days=30,
                                     min_new_public_activity=2):
        """
        Detect temporal blind spots where ideas should have surfaced but didn't.
        
        A blind spot is:
        - High predicted synthesis P_t > θ AND/OR
        - High private activation in topics with small L_m
        - But no public activation within expected window W ≈ L_m + δ
        
        IMPROVED: Only considers public matches within ±recency_window_days of private date.
        If no recent matches, tags as "reactivation_gap" instead of 0-day lag.
        """
        print("\n" + "=" * 80)
        print("🔍 Time-Aware Blind Spot Detection")
        print("=" * 80)
        print(f"   Recency window: ±{recency_window_days} days")
        print(f"   Min new public activity: {min_new_public_activity} items")
        
        if lead_lag_map.empty or topic_ilc.empty:
            return pd.DataFrame()
        
        # Ensure indices are datetime objects
        if not isinstance(activations_priv.index, pd.DatetimeIndex):
            activations_priv.index = pd.to_datetime(activations_priv.index)
        if not isinstance(activations_pub.index, pd.DatetimeIndex):
            activations_pub.index = pd.to_datetime(activations_pub.index)
        
        # Build expected lag map
        expected_lags = dict(zip(
            topic_ilc['private_cluster'],
            topic_ilc['weighted_median_lag']
        ))
        
        blind_spots = []
        
        for date in activations_priv.index:
            # Check each private topic
            for m in activations_priv.columns:
                if m == 'date':
                    continue
                
                priv_activation = activations_priv.loc[date, m]
                
                if priv_activation == 0:
                    continue
                
                # Get expected lag
                expected_lag = expected_lags.get(m, 3.0)  # Default 3 days
                window = expected_lag + 2.0  # Add 2 day buffer
                
                # Check window in future
                # Ensure date is a Timestamp for comparison
                if not isinstance(date, pd.Timestamp):
                    date = pd.to_datetime(date)
                window_start = date
                window_end = date + pd.Timedelta(days=window)
                
                # RECENCY WINDOW: Only consider public matches within ±recency_window_days
                recency_start = date - pd.Timedelta(days=recency_window_days)
                recency_end = date + pd.Timedelta(days=recency_window_days)
                
                # Check if public activation occurred (within expected window AND recency window)
                aligned = lead_lag_map[lead_lag_map['private_cluster'] == m]
                public_clusters = aligned['public_cluster'].unique()
                
                public_activated = False
                recent_public_activated = False
                historical_public_activated = False
                new_public_count = 0
                historical_public_count = 0
                
                for k in public_clusters:
                    if k in activations_pub.columns:
                        # Check within expected window
                        window_activations = activations_pub[
                            (activations_pub.index >= window_start) &
                            (activations_pub.index <= window_end)
                        ][k]
                        window_count = (window_activations > 0).sum()
                        
                        # Check within recency window
                        recent_activations = activations_pub[
                            (activations_pub.index >= recency_start) &
                            (activations_pub.index <= recency_end)
                        ][k]
                        recent_count = (recent_activations > 0).sum()
                        
                        # Check historical (outside recency window)
                        all_activations = activations_pub[k]
                        historical_activations = all_activations[
                            (activations_pub.index < recency_start) | 
                            (activations_pub.index > recency_end)
                        ]
                        historical_count = (historical_activations > 0).sum()
                        
                        if window_count > 0:
                            public_activated = True
                        if recent_count >= min_new_public_activity:
                            recent_public_activated = True
                            new_public_count += recent_count
                        if historical_count > 0:
                            historical_public_activated = True
                            historical_public_count += historical_count
                
                # Determine blind spot type and if it's a blind spot
                is_blind_spot = False
                blind_spot_type = None
                effective_lag = expected_lag
                
                if not public_activated:
                    is_blind_spot = True
                    
                    # Categorize based on recent vs historical matches
                    if historical_public_activated and not recent_public_activated:
                        blind_spot_type = 'reactivation_gap'  # Type A: Dormant public analogue
                        effective_lag = None  # Mark as reactivation gap (inf lag)
                    elif recent_public_activated and new_public_count < min_new_public_activity:
                        blind_spot_type = 'insufficient_activity'
                        effective_lag = expected_lag
                    elif not historical_public_activated:
                        blind_spot_type = 'new_topic'  # No historical precedent
                        effective_lag = expected_lag
                    else:
                        # Multiple private files same day(s) but no collapse
                        if priv_activation >= 2:
                            blind_spot_type = 'burst_no_collapse'  # Type B: Burst→no-collapse
                        else:
                            blind_spot_type = 'channel_mismatch'  # Type C: Research-dense → lightweight
                        effective_lag = expected_lag
                elif not recent_public_activated and historical_public_activated:
                    # Public activation exists but outside recency window
                    is_blind_spot = True
                    blind_spot_type = 'reactivation_gap'
                    effective_lag = None
                
                if is_blind_spot:
                    # Check if synthesis was predicted
                    predicted = False
                    if synthesis_predictions is not None:
                        if date in synthesis_predictions.index:
                            predicted = synthesis_predictions.loc[date] > 0.5
                    
                    # Score by expected hazard (simplified: use ILC)
                    ilc = topic_ilc[topic_ilc['private_cluster'] == m]['ilc_penalized'].values                                                                  
                    expected_hazard = ilc[0] if len(ilc) > 0 else 0.0
                    
                    # Get max similarity to aligned public clusters
                    max_similarity = 0.0
                    if not aligned.empty:
                        max_similarity = aligned['cosine_similarity'].max()
                    
                    blind_spots.append({
                        'date': date,
                        'private_cluster': m,
                        'expected_lag': expected_lag,
                        'effective_lag': effective_lag,
                        'is_reactivation_gap': effective_lag is None,
                        'private_activation': priv_activation,
                        'expected_hazard': expected_hazard,
                        'synthesis_predicted': predicted,
                        'blind_spot_score': expected_hazard * priv_activation,
                        'blind_spot_type': blind_spot_type,
                        'max_similarity': max_similarity,
                        'recent_public_count': new_public_count,
                        'historical_public_count': historical_public_count,
                        'aligned_public_clusters': ','.join([str(k) for k in public_clusters[:5]])  # Top 5
                    })
        
        if not blind_spots:
            print("   No blind spots detected")
            return pd.DataFrame()
        
        df = pd.DataFrame(blind_spots)
        df = df.sort_values('blind_spot_score', ascending=False)
        
        print(f"\n   Detected {len(df)} potential blind spots")
        print(f"   Top 10 blind spot scores:")
        for _, row in df.head(10).iterrows():
            print(f"     {row['date']}: Cluster {row['private_cluster']}, score={row['blind_spot_score']:.4f}")
        
        return df
    
    # Helper methods
    
    def _compute_similarity_matrix(self, centroids_pub, centroids_priv):
        """Compute cosine similarity matrix between centroids."""
        from sklearn.metrics.pairwise import cosine_similarity
        
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
        
        return S
    
    def _get_activation_series(self, activations, cluster_id):
        """Get activation time series for a cluster."""
        if cluster_id in activations.columns:
            return activations[cluster_id]
        return pd.Series(dtype=float)
    
    def _cross_correlation(self, x, y, max_lag):
        """Compute cross-correlation over lag window."""
        # Align indices
        common_dates = x.index.intersection(y.index)
        if len(common_dates) < 10:
            return {'optimal_lag': 0, 'correlation': 0.0, 'max_correlation': 0.0}
        
        x_aligned = x[common_dates].values
        y_aligned = y[common_dates].values
        
        best_corr = -np.inf
        best_lag = 0
        max_corr = 0.0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                x_shifted = x_aligned
                y_shifted = y_aligned
            elif lag > 0:
                if len(x_aligned) <= lag:
                    continue
                x_shifted = x_aligned[:-lag]
                y_shifted = y_aligned[lag:]
            else:
                if len(y_aligned) <= abs(lag):
                    continue
                x_shifted = x_aligned[-lag:]
                y_shifted = y_aligned[:lag]
            
            if len(x_shifted) < 5 or len(y_shifted) < 5:
                continue
            
            min_len = min(len(x_shifted), len(y_shifted))
            x_shifted = x_shifted[:min_len]
            y_shifted = y_shifted[:min_len]
            
            try:
                corr, _ = pearsonr(x_shifted, y_shifted)
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
            except:
                pass
        
        return {
            'optimal_lag': best_lag,
            'correlation': float(best_corr),
            'max_correlation': float(max_corr)
        }
    
    def _compute_transfer_entropy(self, x, y):
        """
        Approximate transfer entropy using correlation.
        Full implementation would use mutual information.
        """
        common = x.index.intersection(y.index)
        if len(common) < 10:
            return 0.0
        
        try:
            corr, _ = pearsonr(x[common], y[common])
            return float(corr ** 2) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _get_event_times(self, activations, cluster_id):
        """Extract event times (dates with non-zero activation)."""
        if cluster_id not in activations.columns:
            return []
        
        series = activations[cluster_id]
        event_times = series[series > 0].index.tolist()
        return [pd.to_datetime(t) for t in event_times]
    
    def _fit_hawkes(self, priv_events, pub_events):
        """
        Fit Hawkes process parameters α and β.
        
        Model: λ(t) = μ + α * Σ β * exp(-β(t - t_i))
        """
        if not priv_events or not pub_events:
            return 0.0, 1.0
        
        # Convert to numeric (days since first event)
        t0 = min(priv_events + pub_events)
        priv_times = [(t - t0).total_seconds() / 86400 for t in priv_events]
        pub_times = [(t - t0).total_seconds() / 86400 for t in pub_events]
        
        # Simple maximum likelihood estimation
        # For simplicity, use moment-based estimator
        if len(pub_times) < 2:
            return 0.0, 1.0
        
        # Estimate β from inter-event intervals
        intervals = np.diff(sorted(pub_times))
        if len(intervals) > 0 and intervals.mean() > 0:
            beta_est = 1.0 / intervals.mean()
        else:
            beta_est = 1.0
        
        # Estimate α from cross-correlation
        # Simplified: use correlation between private and public intensities
        try:
            # Count events in windows
            window_size = 1.0  # 1 day
            max_t = max(priv_times + pub_times)
            bins = np.arange(0, max_t + window_size, window_size)
            
            priv_counts, _ = np.histogram(priv_times, bins=bins)
            pub_counts, _ = np.histogram(pub_times, bins=bins)
            
            if priv_counts.sum() > 0 and pub_counts.sum() > 0:
                corr, _ = pearsonr(priv_counts, pub_counts)
                alpha_est = max(0.0, corr) if not np.isnan(corr) else 0.0
            else:
                alpha_est = 0.0
        except:
            alpha_est = 0.0
        
        return alpha_est, beta_est


class TemporalVisualizer:
    """Visualization utilities for temporal analysis."""
    
    @staticmethod
    def plot_lag_surface(lead_lag_map, top_n=20, save_path=None):
        """Plot heatmap of optimal lags for top cluster pairs."""
        if not MATPLOTLIB_AVAILABLE or lead_lag_map.empty:
            return
        
        top_pairs = lead_lag_map.head(top_n)
        
        # Create pivot table
        pivot = top_pairs.pivot_table(
            values='optimal_lag',
            index='private_cluster',
            columns='public_cluster',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='coolwarm', center=0)
        plt.title('Lead-Lag Surface (Optimal Lag τ*)')
        plt.xlabel('Public Cluster')
        plt.ylabel('Private Cluster')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_latency_histogram(lead_lag_map, save_path=None):
        """Plot distribution of optimal lags."""
        if not MATPLOTLIB_AVAILABLE or lead_lag_map.empty:
            return
        
        lags = lead_lag_map['optimal_lag'].values
        
        plt.figure(figsize=(10, 6))
        plt.hist(lags, bins=range(int(lags.min()), int(lags.max()) + 2), 
                 edgecolor='black', alpha=0.7)
        plt.xlabel('Optimal Lag (days)')
        plt.ylabel('Frequency')
        plt.title('Latency Histogram: Distribution of Lead-Lag Times')
        plt.axvline(x=0, color='r', linestyle='--', label='No lag')
        plt.axvline(x=np.median(lags), color='g', linestyle='--', label=f'Median: {np.median(lags):.1f} days')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_survival_curves(survival_results, save_path=None):
        """Plot Kaplan-Meier survival curves."""
        if not MATPLOTLIB_AVAILABLE or survival_results is None:
            return
        
        kmf = survival_results['kmf']
        
        plt.figure(figsize=(10, 6))
        kmf.plot_survival_function()
        plt.xlabel('Time to Externalization (days)')
        plt.ylabel('Survival Probability')
        plt.title('Survival Curves: Time-to-Public-Externalization')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_lead_lag_braid(activations_priv, activations_pub, 
                            private_cluster, public_cluster,
                            optimal_lag, save_path=None):
        """Plot aligned time series showing lead-lag relationship."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        a_priv = activations_priv[private_cluster] if private_cluster in activations_priv.columns else pd.Series()
        a_pub = activations_pub[public_cluster] if public_cluster in activations_pub.columns else pd.Series()
        
        if a_priv.empty or a_pub.empty:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Shift public series by optimal lag
        if optimal_lag > 0:
            a_pub_shifted = a_pub.shift(-optimal_lag)
        elif optimal_lag < 0:
            a_pub_shifted = a_pub.shift(abs(optimal_lag))
        else:
            a_pub_shifted = a_pub
        
        ax.plot(a_priv.index, a_priv.values, label='Private (Cluster {})'.format(private_cluster), 
                linewidth=2, alpha=0.7)
        ax.plot(a_pub_shifted.index, a_pub_shifted.values, 
                label='Public (Cluster {}) shifted by {} days'.format(public_cluster, optimal_lag),
                linewidth=2, alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Activation')
        ax.set_title('Lead-Lag Braid: Private → Public Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """CLI for advanced temporal analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced temporal analysis for two-pole model'
    )
    parser.add_argument('--lead_lag_map', required=True, help='CSV with lead-lag map')
    parser.add_argument('--activations_pub', required=True, help='CSV with public activations')
    parser.add_argument('--activations_priv', required=True, help='CSV with private activations')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    lead_lag_map = pd.read_csv(args.lead_lag_map)
    activations_pub = pd.read_csv(args.activations_pub, index_col=0, parse_dates=True)
    activations_priv = pd.read_csv(args.activations_priv, index_col=0, parse_dates=True)
    
    # Run analysis
    analyzer = AdvancedTemporalAnalyzer()
    
    # Per-topic learning lag
    topic_ilc = analyzer.compute_per_topic_learning_lag(lead_lag_map)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not topic_ilc.empty:
        topic_ilc.to_csv(output_dir / 'topic_ilc.csv', index=False)
        print(f"\n✅ Saved topic ILC to {output_dir / 'topic_ilc.csv'}")


if __name__ == "__main__":
    main()
