#!/usr/bin/env python3
"""
Step 5: THRML Prototype - Idea Domain Formation

Maps daily k-NN graph to Potts-style EBM and watches domains crystallize
as temperature (entropy) and field (exploration pressure) change.

Energy: E(s) = -γ * Σ w_ij * 1[s_i = s_j] - Σ <h_i, e_s_i>

Where:
- Nodes: categorical variables (theme labels)
- Edges: coupling weights (cosine similarity)
- Temperature: β = 1/Entropy
- External field: h_i = α * ExplorationPressure
- Coupling scale: γ = IntegrationPotential
"""
import json
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Try to import THRML
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    print("✓ JAX available")
except ImportError:
    HAS_JAX = False
    print("⚠️  JAX not available. Using NumPy-only simulation.")

try:
    from thrml.block_management import Block
    from thrml.block_sampling import BlockGibbsSpec, sample_states, SamplingSchedule
    from thrml.pgm import CategoricalNode
    from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
    from thrml.factor import FactorSamplingProgram
    HAS_THRML = True
    print("✓ THRML available")
except ImportError:
    HAS_THRML = False
    print("⚠️  THRML not available. Using simplified Gibbs sampler.")


def load_knn_graph(nodes_file, edges_file):
    """Load k-NN graph from CSV files."""
    print("\n📥 Loading k-NN graph...")
    
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    # Build NetworkX graph
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['id'], text=row.get('text', ''))
    
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=float(row['similarity']))
    
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    return G, nodes_df, edges_df


def simple_gibbs_sampler(G, K=6, beta=1.0, gamma=0.8, alpha=1.0, exploration_pressure=0.25, n_steps=100):
    """
    Simple Gibbs sampler for Potts model (fallback if THRML not available).
    
    Energy: E(s) = -γ * Σ w_ij * 1[s_i = s_j] - Σ h_i * s_i
    """
    print(f"\n🔄 Running Gibbs Sampler (K={K}, β={beta:.3f}, γ={gamma:.3f}, α={alpha:.3f})...")
    
    n_nodes = G.number_of_nodes()
    node_list = list(G.nodes())
    id2idx = {node: i for i, node in enumerate(node_list)}
    
    # Initialize random states
    states = np.random.randint(0, K, size=n_nodes)
    
    # Build adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))
    weights = {}
    for u, v, data in G.edges(data=True):
        i, j = id2idx[u], id2idx[v]
        w = data.get('weight', 1.0)
        adj_matrix[i, j] = w
        adj_matrix[j, i] = w
        weights[(i, j)] = w
        weights[(j, i)] = w
    
    # External field (bias toward theme 0, scaled by exploration pressure)
    h = np.ones((n_nodes, K)) / K + alpha * exploration_pressure * np.eye(K)[0]
    
    # Gibbs sampling
    mixing_time = 0
    stable = False
    
    for step in range(n_steps):
        old_states = states.copy()
        
        # Update each node
        for i in range(n_nodes):
            # Calculate energy for each possible state
            energies = np.zeros(K)
            
            for s in range(K):
                # Coupling term: -γ * Σ w_ij * 1[s_i = s_j]
                coupling_energy = 0.0
                for j in range(n_nodes):
                    if adj_matrix[i, j] > 0:
                        w_ij = adj_matrix[i, j]
                        if states[j] == s:
                            coupling_energy += gamma * w_ij
                
                # Field term: -Σ h_i * s_i
                field_energy = h[i, s]
                
                # Total energy (negative for log-probability)
                energies[s] = -beta * coupling_energy - field_energy
            
            # Convert to probabilities (softmax)
            exp_energies = np.exp(-energies)
            probs = exp_energies / exp_energies.sum()
            
            # Sample new state
            states[i] = np.random.choice(K, p=probs)
        
        # Check for stability
        if step > 10 and not stable:
            if np.all(states == old_states):
                mixing_time = step
                stable = True
    
    if not stable:
        mixing_time = n_steps
    
    return states, mixing_time, stable


def thrml_sampler(G, K=6, beta=1.0, gamma=0.8, alpha=1.0, exploration_pressure=0.25):
    """
    THRML-based sampler (if available).
    """
    if not HAS_THRML:
        return None, None, False
    
    print(f"\n🔄 Running THRML Sampler (K={K}, β={beta:.3f}, γ={gamma:.3f}, α={alpha:.3f})...")
    
    node_list = list(G.nodes())
    id2idx = {n: i for i, n in enumerate(node_list)}
    
    # Categorical variables per node
    nodes_dict = {n: CategoricalNode() for n in node_list}
    for n in G.nodes():
        G.nodes[n]['node'] = nodes_dict[n]
    
    # Edges as parallel arrays
    u = jnp.array([id2idx[a] for a, b in G.edges()])
    v = jnp.array([id2idx[b] for a, b in G.edges()])
    w = jnp.array([G[a][b].get('weight', 1.0) for a, b in G.edges()])
    
    # Potts coupling
    Wedge = beta * gamma * jnp.einsum("e,ab->eab", w, jnp.eye(K))
    pair_factor = CategoricalEBMFactor([Block(u), Block(v)], Wedge)
    
    # External field
    h = jnp.ones((len(node_list), K)) / K + alpha * exploration_pressure * jnp.eye(K)[0]
    unary_factor = CategoricalEBMFactor([Block(jnp.arange(len(node_list)))],
                                        jnp.expand_dims(h, 0))
    
    # 2-coloring for parallel blocks
    colors = {n: c % 2 for c, n in enumerate(node_list)}
    blocks = [Block(jnp.array([id2idx[n] for n, c in colors.items() if c == col])) 
              for col in (0, 1)]
    spec = BlockGibbsSpec(blocks, [])
    sampler = CategoricalGibbsConditional(K)
    
    prog = FactorSamplingProgram(spec, [sampler] * len(spec.free_blocks),
                                 interactions=[pair_factor, unary_factor], controls=[])
    
    # Sampling
    key = jax.random.key(0)
    init = [jax.random.randint(key, (1, len(b.nodes)), 0, K, dtype=jnp.uint8) 
            for b in spec.free_blocks]
    schedule = SamplingSchedule(n_warmup=10, n_samples=100, steps_per_sample=5)
    
    def samples_fn(init, key):
        return sample_states(key, prog, schedule, init, [], 
                            [Block(list(range(len(node_list))))])
    
    samples_fn_jit = jax.jit(samples_fn)
    states_array = samples_fn_jit(init, key)[0]
    states = states_array[-1]  # Final state
    
    # Estimate mixing time (simplified)
    mixing_time = schedule.n_warmup + schedule.n_samples * schedule.steps_per_sample
    
    return np.array(states), mixing_time, True


def simulate_daily_graph(date, daily_dir, entropy, integration, exploration_pressure, K=6):
    """
    Simulate idea domain formation for a single day.
    """
    nodes_file = Path(daily_dir) / "knn_nodes.csv"
    edges_file = Path(daily_dir) / "knn_edges.csv"
    
    if not nodes_file.exists() or not edges_file.exists():
        return None
    
    # Load graph
    G, nodes_df, edges_df = load_knn_graph(nodes_file, edges_file)
    
    # Parameters
    beta = 1.0 / entropy if entropy > 0 else 1.0  # Temperature
    gamma = integration  # Coupling strength
    alpha = 1.0  # Field scaling
    exploration_pressure = exploration_pressure
    
    # Run sampler (use simple Gibbs sampler for now - more reliable)
    # TODO: Fix THRML API usage for full implementation
    states, mixing_time, stable = simple_gibbs_sampler(G, K, beta, gamma, alpha, exploration_pressure)
    
    if states is None:
        return None
    
    # Calculate diagnostics
    counts = Counter(states)
    counts_array = np.array([counts.get(i, 0) for i in range(K)])
    total = len(states)
    
    if total > 0:
        domain_frac = counts_array.max() / total  # Largest domain fraction
        probs = (counts_array + 1e-9) / total
        label_entropy = -np.sum(probs * np.log(probs))  # Label entropy
    else:
        domain_frac = 0.0
        label_entropy = 0.0
    
    return {
        'date': date,
        'domain_frac': float(domain_frac),
        'label_entropy': float(label_entropy),
        'mixing_time': int(mixing_time),
        'stable': bool(stable),
        'n_nodes': len(states),
        'states': states.tolist() if isinstance(states, np.ndarray) else states,
    }


def compare_sim_to_reality(sim_results, iv_df):
    """
    Compare simulation outputs to real metrics.
    Fit linear model: (domain_frac, label_entropy) → (ER, posts, IV, ΔIV)
    """
    print("\n🔬 Comparing Simulation to Reality...")
    
    # Prepare data
    sim_df = pd.DataFrame(sim_results)
    
    # Merge with real metrics
    iv_df['date'] = pd.to_datetime(iv_df['date']).dt.strftime('%Y-%m-%d')
    sim_df['date'] = sim_df['date'].astype(str)
    merged = sim_df.merge(iv_df, on='date', how='inner')
    
    if len(merged) == 0:
        print("  ⚠️  No matching dates found")
        return None
    
    print(f"  Matching days: {len(merged)}")
    
    # Fit models: sim → real
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    results = {}
    
    targets = {
        'entropy_reduction': merged['entropy_reduction'].values,
        'posts': merged['posts'].values,
        'insight_velocity': merged['insight_velocity'].values,
        'delta_iv': merged['delta_iv'].values,
    }
    
    features = merged[['domain_frac', 'label_entropy']].values
    
    print("\n  Fits:")
    for target_name, target_values in targets.items():
        model = LinearRegression()
        model.fit(features, target_values)
        predictions = model.predict(features)
        r2 = r2_score(target_values, predictions)
        
        results[target_name] = {
            'r2': r2,
            'coef_domain_frac': model.coef_[0],
            'coef_label_entropy': model.coef_[1],
            'intercept': model.intercept_,
        }
        
        print(f"    {target_name}:")
        print(f"      R² = {r2:.4f}")
        print(f"      Domain frac coefficient: {model.coef_[0]:+.4f}")
        print(f"      Label entropy coefficient: {model.coef_[1]:+.4f}")
    
    return results, merged


def create_simulation_plots(sim_results, merged_df, output_dir="."):
    """Create plots comparing simulation to reality."""
    print("\n📊 Creating Simulation Comparison Plots...")
    
    if merged_df is None or len(merged_df) == 0:
        print("  ⚠️  No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    dates = pd.to_datetime(merged_df['date'])
    
    # Plot 1: Domain Fraction vs Real Metrics
    ax1 = axes[0, 0]
    ax1.plot(dates, merged_df['domain_frac'], 'o-', label='Sim: Domain Fraction',
            linewidth=2, markersize=8)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates, merged_df['entropy_reduction'], 's--', color='red',
                  label='Real: Entropy Reduction', linewidth=2, markersize=6)
    ax1.set_ylabel('Domain Fraction', fontsize=11, fontweight='bold')
    ax1_twin.set_ylabel('Entropy Reduction', fontsize=11, fontweight='bold', color='red')
    ax1.set_title('Domain Formation vs Synthesis', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Label Entropy vs Real Metrics
    ax2 = axes[0, 1]
    ax2.plot(dates, merged_df['label_entropy'], 'o-', label='Sim: Label Entropy',
            linewidth=2, markersize=8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(dates, merged_df['insight_velocity'], 's--', color='purple',
                  label='Real: IV', linewidth=2, markersize=6)
    ax2.set_ylabel('Label Entropy', fontsize=11, fontweight='bold')
    ax2_twin.set_ylabel('Insight Velocity', fontsize=11, fontweight='bold', color='purple')
    ax2.set_title('Label Entropy vs IV', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Domain Fraction vs Posts
    ax3 = axes[1, 0]
    ax3.scatter(merged_df['domain_frac'], merged_df['posts'], 
               s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
    ax3.set_xlabel('Domain Fraction (Sim)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Posts (Real)', fontsize=11, fontweight='bold')
    ax3.set_title('Domain Formation → Synthesis', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Label Entropy vs Entropy Reduction
    ax4 = axes[1, 1]
    ax4.scatter(merged_df['label_entropy'], merged_df['entropy_reduction'],
               s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
    ax4.set_xlabel('Label Entropy (Sim)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Entropy Reduction (Real)', fontsize=11, fontweight='bold')
    ax4.set_title('Entropy Reduction Correlation', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis for time plots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    sim_file = Path(output_dir) / "simulation_reality_comparison.png"
    plt.savefig(sim_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot to: {sim_file}")
    plt.close()


def main():
    daily_analysis_dir = Path("weekly_analysis")
    iv_file = "iv_metrics_calibrated.csv"
    output_dir = Path(".")
    
    print("=" * 80)
    print("⚛️  Step 5: THRML Prototype - Idea Domain Formation")
    print("=" * 80)
    
    # Load IV metrics
    print("\n📥 Loading IV metrics...")
    iv_df = pd.read_csv(iv_file)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    
    print(f"  Loaded metrics for {len(iv_df)} days")
    
    # Simulate each day
    print("\n🔄 Simulating Idea Domain Formation...")
    sim_results = []
    
    for _, row in iv_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        date_dir = daily_analysis_dir / date_str
        
        if not date_dir.exists():
            print(f"  ⚠️  Skipping {date_str} (no data)")
            continue
        
        print(f"\n  Simulating {date_str}...")
        result = simulate_daily_graph(
            date_str,
            date_dir,
            entropy=row['breadth'],
            integration=row['integration'],
            exploration_pressure=row['exploration_pressure']
        )
        
        if result:
            sim_results.append(result)
            print(f"    Domain fraction: {result['domain_frac']:.4f}")
            print(f"    Label entropy: {result['label_entropy']:.4f}")
            print(f"    Mixing time: {result['mixing_time']}")
    
    if len(sim_results) == 0:
        print("\n⚠️  No simulation results. Check if daily analysis directories exist.")
        return
    
    # Save simulation results
    sim_file = "thrml_simulation_results.json"
    with open(sim_file, 'w') as f:
        json.dump(sim_results, f, indent=2)
    print(f"\n✓ Saved simulation results to: {sim_file}")
    
    # Compare to reality
    fit_results, merged_df = compare_sim_to_reality(sim_results, iv_df)
    
    # Save merged results
    if merged_df is not None:
        merged_file = "sim_reality_merged.csv"
        merged_df.to_csv(merged_file, index=False)
        print(f"✓ Saved merged results to: {merged_file}")
    
    # Create visualizations
    try:
        create_simulation_plots(sim_results, merged_df, output_dir)
    except Exception as e:
        print(f"\n⚠️  Error creating plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SIMULATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Date':<12} {'Domain Frac':<12} {'Label Entropy':<15} {'Mixing Time':<12}")
    print("-" * 80)
    
    for result in sim_results:
        print(f"{result['date']:<12} {result['domain_frac']:<12.4f} {result['label_entropy']:<15.4f} {result['mixing_time']:<12}")
    
    print("\n" + "=" * 80)
    print("✅ THRML simulation complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {sim_file} - Simulation results")
    if merged_df is not None:
        print(f"  - {merged_file} - Merged simulation and reality data")
        print(f"  - simulation_reality_comparison.png - Comparison plots")


if __name__ == "__main__":
    main()

