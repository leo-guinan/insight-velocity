#!/usr/bin/env python3
"""
Step 4: Dynamic System Modeling - Insight Thermodynamics

Models IV as a dynamical system:
- IV_{t+1} = IV_t + ΔIV_t
- ΔIV_{t+1} = f(ExplorationPressure_t, Entropy_t, Novelty_t)

Treated as a damped driven oscillator:
- m*x'' + c*x' + k*x = F(t)

Where:
- x = IV (position)
- x' = ΔIV (velocity)
- F(t) = Exploration Pressure (driving force)
- c = (1 - Integration Potential) (damping)
- k = (1 - Compression Readiness) (elasticity)
"""
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path


def fit_delta_iv_model(iv_df):
    """
    Fit a model for ΔIV based on features:
    ΔIV_{t+1} = f(ExplorationPressure_t, Entropy_t, Novelty_t)
    """
    print("\n🔬 Fitting ΔIV Model...")
    
    # Prepare features
    features = ['exploration_pressure', 'breadth', 'novelty', 'integration', 'compression']
    X = iv_df[features].values
    y = iv_df['delta_iv'].values
    
    # Fit linear model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Predictions
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"  Model R²: {r2:.4f}")
    print(f"  Model MSE: {mse:.6f}")
    print(f"\n  Feature Coefficients:")
    for i, feature in enumerate(features):
        print(f"    {feature}: {model.coef_[i]:+.4f}")
    print(f"    Intercept: {model.intercept_:+.4f}")
    
    return model, scaler, features


def simulate_oscillator_model(iv_df, model, scaler, features):
    """
    Simulate IV using oscillator model:
    m*x'' + c*x' + k*x = F(t)
    
    Where:
    - x = IV (position)
    - x' = ΔIV (velocity)
    - F(t) = Exploration Pressure
    - c = (1 - Integration Potential) (damping)
    - k = (1 - Compression Readiness) (elasticity)
    """
    print("\n🎯 Simulating Oscillator Model...")
    
    # Initialize simulation
    sim_df = iv_df.copy()
    sim_df['iv_simulated'] = 0.0
    sim_df['delta_iv_simulated'] = 0.0
    sim_df['iv_simulated'][0] = sim_df['insight_velocity'].iloc[0]
    sim_df['delta_iv_simulated'][0] = sim_df['delta_iv'].iloc[0]
    
    # Simulate forward
    for i in range(1, len(sim_df)):
        # Current state
        iv_current = sim_df['iv_simulated'].iloc[i-1]
        delta_iv_current = sim_df['delta_iv_simulated'].iloc[i-1]
        
        # Parameters
        exploration_pressure = sim_df['exploration_pressure'].iloc[i]
        integration = sim_df['integration'].iloc[i]
        compression = sim_df['compression'].iloc[i]
        
        # Oscillator parameters
        c = 1.0 - integration  # Damping coefficient
        k = 1.0 - compression   # Elasticity coefficient
        F_t = exploration_pressure  # Driving force
        
        # Simple Euler integration for oscillator
        # m*x'' + c*x' + k*x = F(t)
        # For simplicity, assume m=1
        # x'' = F(t) - c*x' - k*x
        
        dt = 1.0  # Daily timestep
        acceleration = F_t - c * delta_iv_current - k * iv_current
        
        # Update velocity and position
        delta_iv_new = delta_iv_current + acceleration * dt
        iv_new = iv_current + delta_iv_new * dt
        
        sim_df.loc[sim_df.index[i], 'iv_simulated'] = iv_new
        sim_df.loc[sim_df.index[i], 'delta_iv_simulated'] = delta_iv_new
    
    # Calculate error
    mse = mean_squared_error(sim_df['insight_velocity'], sim_df['iv_simulated'])
    print(f"  Simulation MSE: {mse:.6f}")
    print(f"  Average IV error: {np.mean(np.abs(sim_df['insight_velocity'] - sim_df['iv_simulated'])):.6f}")
    
    return sim_df


def predict_forward(iv_df, model, scaler, features, n_days=5):
    """
    Predict IV forward in time using the model.
    """
    print(f"\n🔮 Forward Prediction ({n_days} days)...")
    
    # Use last day's features as baseline
    last_day = iv_df.iloc[-1].copy()
    
    predictions = []
    current_iv = last_day['insight_velocity']
    current_delta_iv = last_day['delta_iv']
    
    for day in range(1, n_days + 1):
        # Simple forward projection (could be improved with trend extrapolation)
        features_current = np.array([[
            last_day['exploration_pressure'],
            last_day['breadth'],
            last_day['novelty'],
            last_day['integration'],
            last_day['compression']
        ]])
        
        features_scaled = scaler.transform(features_current)
        predicted_delta_iv = model.predict(features_scaled)[0]
        
        # Update IV
        predicted_iv = current_iv + predicted_delta_iv
        
        predictions.append({
            'day': day,
            'date_projected': f"+{day}",
            'iv_predicted': predicted_iv,
            'delta_iv_predicted': predicted_delta_iv,
        })
        
        current_iv = predicted_iv
        current_delta_iv = predicted_delta_iv
    
    pred_df = pd.DataFrame(predictions)
    print(f"  Predicted IV after {n_days} days: {pred_df['iv_predicted'].iloc[-1]:.4f}")
    
    return pred_df


def create_thermodynamics_plot(iv_df, output_dir="."):
    """
    Create Insight Thermodynamics plot:
    - x-axis = Entropy (temperature)
    - y-axis = IV (energy)
    - Color = Momentum
    """
    print("\n📊 Creating Insight Thermodynamics Plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data
    entropy = iv_df['breadth']  # Temperature
    energy = iv_df['insight_velocity']  # Energy
    momentum = iv_df['iv_momentum']  # Color
    
    # Color mapping
    colors = momentum
    sizes = np.abs(momentum) * 500 + 100  # Size by absolute momentum
    
    # Scatter plot
    scatter = ax.scatter(entropy, energy, s=sizes, c=colors,
                        cmap='RdYlGn', alpha=0.7,
                        edgecolors='black', linewidths=1.5,
                        vmin=momentum.min(), vmax=momentum.max())
    
    # Annotate points with dates
    for idx, row in iv_df.iterrows():
        date_short = row['date'].strftime('%m/%d') if isinstance(row['date'], pd.Timestamp) else '/'.join(row['date'].split('-')[-2:])
        ax.annotate(date_short,
                   (row['breadth'], row['insight_velocity']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Mark phase transitions
    # Eruption phase (high entropy, high exploration pressure)
    eruption_mask = iv_df['eruption_detected']
    if eruption_mask.any():
        ax.scatter(iv_df.loc[eruption_mask, 'breadth'],
                  iv_df.loc[eruption_mask, 'insight_velocity'],
                  s=300, marker='^', c='red', label='Eruption',
                  edgecolors='black', linewidths=2, zorder=5)
    
    # Synthesis phase (posts)
    synthesis_mask = iv_df['posts'] > 0
    if synthesis_mask.any():
        ax.scatter(iv_df.loc[synthesis_mask, 'breadth'],
                  iv_df.loc[synthesis_mask, 'insight_velocity'],
                  s=300, marker='*', c='darkgreen', label='Synthesis',
                  edgecolors='black', linewidths=2, zorder=5)
    
    # Add phase zones
    # High temperature (exploration) zone
    ax.axvspan(0.95, 1.0, alpha=0.1, color='red', label='High Temperature Zone')
    # Low temperature (synthesis) zone
    ax.axvspan(0.85, 0.90, alpha=0.1, color='green', label='Synthesis Zone')
    
    # Labels and title
    ax.set_xlabel('Entropy (Temperature β⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('IV (Energy)', fontsize=12, fontweight='bold')
    ax.set_title('Insight Thermodynamics\nEntropy vs Energy (colored by Momentum)', 
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('IV Momentum', fontsize=11, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best')
    
    # Add interpretation text
    ax.text(0.02, 0.98,
           'Phase Transitions:\n'
           '• Eruption: High temp, high energy\n'
           '• Synthesis: Low temp, stable energy',
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    thermo_file = Path(output_dir) / "insight_thermodynamics.png"
    plt.savefig(thermo_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved thermodynamics plot to: {thermo_file}")
    plt.close()


def create_simulation_plots(sim_df, output_dir="."):
    """Create plots comparing actual vs simulated IV."""
    print("\n📈 Creating Simulation Comparison Plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    dates = pd.to_datetime(sim_df['date'])
    
    # Plot 1: IV Comparison
    ax1 = axes[0]
    ax1.plot(dates, sim_df['insight_velocity'], 'o-', label='Actual IV',
            linewidth=2, markersize=8, color='blue')
    ax1.plot(dates, sim_df['iv_simulated'], 's--', label='Simulated IV (Oscillator)',
            linewidth=2, markersize=8, color='red', alpha=0.7)
    
    ax1.set_ylabel('Insight Velocity', fontsize=11, fontweight='bold')
    ax1.set_title('Oscillator Model: Actual vs Simulated IV', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ΔIV Comparison
    ax2 = axes[1]
    ax2.plot(dates, sim_df['delta_iv'], 'o-', label='Actual ΔIV',
            linewidth=2, markersize=8, color='purple')
    ax2.plot(dates, sim_df['delta_iv_simulated'], 's--', label='Simulated ΔIV',
            linewidth=2, markersize=8, color='orange', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ΔIV (Acceleration)', fontsize=11, fontweight='bold')
    ax2.set_title('Oscillator Model: Actual vs Simulated Acceleration', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    sim_file = Path(output_dir) / "oscillator_simulation.png"
    plt.savefig(sim_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved simulation plot to: {sim_file}")
    plt.close()


def calculate_energy_efficiency(iv_df):
    """
    Calculate energy efficiency = (IV gain per unit Exploration Pressure)
    """
    print("\n⚡ Calculating Energy Efficiency...")
    
    # Energy efficiency = ΔIV / Exploration Pressure
    iv_df['energy_efficiency'] = iv_df['delta_iv'] / (iv_df['exploration_pressure'] + 1e-10)
    
    # Alternative: IV gain per unit exploration pressure over window
    window = 3
    iv_df['rolling_efficiency'] = (
        iv_df['delta_iv'].rolling(window=window, min_periods=1).sum() /
        (iv_df['exploration_pressure'].rolling(window=window, min_periods=1).sum() + 1e-10)
    )
    
    print(f"  Average energy efficiency: {iv_df['energy_efficiency'].mean():.4f}")
    print(f"  Average rolling efficiency: {iv_df['rolling_efficiency'].mean():.4f}")
    
    return iv_df


def generate_system_report(iv_df, sim_df, pred_df, output_file="dynamic_system_report.txt"):
    """Generate a report on the dynamical system model."""
    print("\n📝 Generating System Report...")
    
    report = []
    report.append("=" * 80)
    report.append("⚙️  DYNAMIC SYSTEM MODELING REPORT - Insight Thermodynamics")
    report.append("=" * 80)
    report.append("")
    
    # Model summary
    report.append("📊 MODEL SUMMARY")
    report.append("-" * 80)
    report.append(f"Model Type: Damped Driven Oscillator")
    report.append(f"Equation: m*x'' + c*x' + k*x = F(t)")
    report.append("")
    report.append("Variables:")
    report.append("  - x = IV (position in insight space)")
    report.append("  - x' = ΔIV (velocity / creative acceleration)")
    report.append("  - F(t) = Exploration Pressure (driving force)")
    report.append("  - c = (1 - Integration Potential) (damping)")
    report.append("  - k = (1 - Compression Readiness) (elasticity)")
    report.append("")
    
    # Simulation accuracy
    report.append("=" * 80)
    report.append("🎯 SIMULATION ACCURACY")
    report.append("=" * 80)
    mse = mean_squared_error(sim_df['insight_velocity'], sim_df['iv_simulated'])
    mae = np.mean(np.abs(sim_df['insight_velocity'] - sim_df['iv_simulated']))
    report.append(f"Mean Squared Error: {mse:.6f}")
    report.append(f"Mean Absolute Error: {mae:.6f}")
    report.append("")
    
    # Phase transitions
    report.append("=" * 80)
    report.append("🔄 PHASE TRANSITIONS")
    report.append("=" * 80)
    report.append("")
    
    eruptions = iv_df[iv_df['eruption_detected']]
    if len(eruptions) > 0:
        report.append("🔥 Eruption Phase (High Temperature, High Energy):")
        for _, eruption in eruptions.iterrows():
            date_str = eruption['date'].strftime('%Y-%m-%d') if isinstance(eruption['date'], pd.Timestamp) else eruption['date']
            report.append(f"  {date_str}:")
            report.append(f"    Temperature (Entropy): {eruption['breadth']:.4f}")
            report.append(f"    Energy (IV): {eruption['insight_velocity']:.4f}")
            report.append(f"    Driving Force (Exploration Pressure): {eruption['exploration_pressure']:.4f}")
            report.append(f"    Damping (1 - Integration): {1.0 - eruption['integration']:.4f}")
            report.append(f"    Elasticity (1 - Compression): {1.0 - eruption['compression']:.4f}")
        report.append("")
    
    synthesis = iv_df[iv_df['posts'] > 0]
    if len(synthesis) > 0:
        report.append("✍️  Synthesis Phase (Low Temperature, Stable Energy):")
        for _, synth in synthesis.iterrows():
            date_str = synth['date'].strftime('%Y-%m-%d') if isinstance(synth['date'], pd.Timestamp) else synth['date']
            report.append(f"  {date_str}:")
            report.append(f"    Temperature (Entropy): {synth['breadth']:.4f}")
            report.append(f"    Energy (IV): {synth['insight_velocity']:.4f}")
            report.append(f"    Posts: {int(synth['posts'])}")
        report.append("")
    
    # Energy efficiency
    report.append("=" * 80)
    report.append("⚡ ENERGY EFFICIENCY")
    report.append("=" * 80)
    report.append(f"Average: {iv_df['energy_efficiency'].mean():.4f}")
    report.append(f"Best: {iv_df['energy_efficiency'].max():.4f} ({iv_df.loc[iv_df['energy_efficiency'].idxmax(), 'date']})")
    report.append("")
    
    # Forward predictions
    if len(pred_df) > 0:
        report.append("=" * 80)
        report.append("🔮 FORWARD PREDICTIONS")
        report.append("=" * 80)
        for _, pred in pred_df.iterrows():
            report.append(f"  Day {pred['day']}: IV = {pred['iv_predicted']:.4f} (ΔIV = {pred['delta_iv_predicted']:+.4f})")
        report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved report to: {output_file}")
    
    return report_text


def main():
    iv_file = "iv_metrics_temporal.csv"
    output_dir = Path(".")
    
    print("=" * 80)
    print("⚙️  Step 4: Dynamic System Modeling - Insight Thermodynamics")
    print("=" * 80)
    
    # Load temporal metrics
    print("\n📥 Loading temporal metrics...")
    iv_df = pd.read_csv(iv_file)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    
    print(f"  Loaded metrics for {len(iv_df)} days")
    
    # Fit ΔIV model
    model, scaler, features = fit_delta_iv_model(iv_df)
    
    # Simulate oscillator model
    sim_df = simulate_oscillator_model(iv_df, model, scaler, features)
    
    # Predict forward
    pred_df = predict_forward(iv_df, model, scaler, features, n_days=5)
    
    # Calculate energy efficiency
    iv_df = calculate_energy_efficiency(iv_df)
    sim_df = calculate_energy_efficiency(sim_df)
    
    # Save enhanced metrics
    enhanced_file = "iv_metrics_dynamic.csv"
    iv_df.to_csv(enhanced_file, index=False)
    print(f"\n✓ Saved enhanced metrics to: {enhanced_file}")
    
    # Create visualizations
    try:
        create_thermodynamics_plot(iv_df, output_dir)
        create_simulation_plots(sim_df, output_dir)
    except Exception as e:
        print(f"\n⚠️  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate report
    report_text = generate_system_report(iv_df, sim_df, pred_df)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 DYNAMIC SYSTEM MODELING SUMMARY")
    print("=" * 80)
    print(f"\n{'Date':<12} {'IV':<8} {'IV_sim':<8} {'Error':<8} {'Efficiency':<12}")
    print("-" * 80)
    
    for _, row in sim_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
        error = abs(row['insight_velocity'] - row['iv_simulated'])
        print(f"{date_str:<12} {row['insight_velocity']:<8.4f} {row['iv_simulated']:<8.4f} "
              f"{error:<8.4f} {row['energy_efficiency']:<12.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Dynamic system modeling complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {enhanced_file} - Enhanced metrics with system dynamics")
    print(f"  - insight_thermodynamics.png - Thermodynamics phase plot")
    print(f"  - oscillator_simulation.png - Model simulation comparison")
    print(f"  - dynamic_system_report.txt - Detailed system analysis")


if __name__ == "__main__":
    main()

