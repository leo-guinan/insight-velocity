#!/usr/bin/env python3
"""
Step 5: Backtest + Calibrate

1. Target: days with posts (synthesis=1) vs none (0)
2. Evaluate: precision/recall, ROC-AUC, PR-AUC, Brier score for synthesis probability
3. Reweight IV components using logistic regression/XGBoost
4. Granger causality tests
"""
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, 
    average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Using Logistic Regression only.")


def prepare_targets(iv_df):
    """Prepare binary target: synthesis=1 if posts>0, else 0."""
    print("\n🎯 Preparing Targets...")
    
    iv_df = iv_df.copy()
    iv_df['synthesis'] = (iv_df['posts'] > 0).astype(int)
    
    synthesis_count = iv_df['synthesis'].sum()
    total_count = len(iv_df)
    
    print(f"  Synthesis days: {synthesis_count}/{total_count} ({synthesis_count/total_count:.1%})")
    print(f"  Non-synthesis days: {total_count - synthesis_count}/{total_count}")
    
    return iv_df


def evaluate_synthesis_forecast(iv_df):
    """
    Evaluate the existing synthesis probability forecast.
    Metrics: precision/recall, ROC-AUC, PR-AUC, Brier score
    """
    print("\n📊 Evaluating Synthesis Forecast...")
    
    y_true = iv_df['synthesis']
    y_pred_prob = iv_df['synthesis_probability']
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_pred_prob)
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Brier Score: {brier:.4f} (lower is better)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"    False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    return {
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier': brier,
        'confusion_matrix': cm
    }


def train_synthesis_model(iv_df, use_xgboost=False):
    """
    Train a model to predict synthesis from features.
    Reweights IV components based on learned weights.
    """
    print("\n🔬 Training Synthesis Prediction Model...")
    
    # Features for synthesis prediction
    features = ['breadth', 'novelty', 'delta_iv', 'iv_momentum', 'exploration_pressure',
                'integration', 'compression', 'in_sweet_spot']
    
    # Filter available features
    available_features = [f for f in features if f in iv_df.columns]
    
    X = iv_df[available_features].values
    y = iv_df['synthesis'].values
    
    print(f"  Features: {available_features}")
    print(f"  Samples: {len(X)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    if use_xgboost and HAS_XGBOOST:
        print("  Using XGBoost...")
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        print("  Using Logistic Regression...")
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_scaled, y)
    
    # Predictions
    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)
    
    # Evaluate
    metrics = {
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_prob) if len(np.unique(y)) > 1 else 0.0,
        'pr_auc': average_precision_score(y, y_pred_prob) if len(np.unique(y)) > 1 else 0.0,
        'brier': brier_score_loss(y, y_pred_prob),
    }
    
    print(f"\n  Model Performance:")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"    Brier Score: {metrics['brier']:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = None
    
    if importances is not None:
        print(f"\n  Feature Importances:")
        for i, feature in enumerate(available_features):
            print(f"    {feature}: {importances[i]:.4f}")
    
    # Add predictions to dataframe
    iv_df['synthesis_prob_learned'] = y_pred_prob
    iv_df['synthesis_pred_learned'] = y_pred
    
    return model, scaler, available_features, metrics


def granger_causality_test(iv_df, max_lag=2):
    """
    Test if Outliers/Entropy "cause" IV (Granger causality).
    Keep lags that are predictive.
    """
    print("\n🔍 Granger Causality Tests...")
    
    from scipy.stats import pearsonr
    
    # Prepare series
    entropy = iv_df['breadth'].values
    outliers = iv_df['outlier_fraction'].values
    iv = iv_df['insight_velocity'].values
    delta_iv = iv_df['delta_iv'].values
    
    results = {}
    
    # Test lags 1-2
    for lag in range(1, max_lag + 1):
        if len(iv_df) > lag:
            # Lag entropy
            entropy_lagged = entropy[:-lag]
            iv_current = iv[lag:]
            
            if len(entropy_lagged) > 1:
                corr_entropy, p_entropy = pearsonr(entropy_lagged, iv_current)
                results[f'entropy_lag_{lag}'] = {
                    'correlation': corr_entropy,
                    'p_value': p_entropy,
                    'significant': p_entropy < 0.05
                }
                print(f"  Entropy (lag {lag}) → IV: r={corr_entropy:.4f}, p={p_entropy:.4f} {'*' if p_entropy < 0.05 else ''}")
            
            # Lag outliers
            outliers_lagged = outliers[:-lag]
            if len(outliers_lagged) > 1:
                corr_outliers, p_outliers = pearsonr(outliers_lagged, iv_current)
                results[f'outliers_lag_{lag}'] = {
                    'correlation': corr_outliers,
                    'p_value': p_outliers,
                    'significant': p_outliers < 0.05
                }
                print(f"  Outliers (lag {lag}) → IV: r={corr_outliers:.4f}, p={p_outliers:.4f} {'*' if p_outliers < 0.05 else ''}")
            
            # Lag outliers → ΔIV
            delta_iv_current = delta_iv[lag:]
            if len(outliers_lagged) == len(delta_iv_current) and len(outliers_lagged) > 1:
                corr_delta, p_delta = pearsonr(outliers_lagged, delta_iv_current)
                results[f'outliers_lag_{lag}_delta_iv'] = {
                    'correlation': corr_delta,
                    'p_value': p_delta,
                    'significant': p_delta < 0.05
                }
                print(f"  Outliers (lag {lag}) → ΔIV: r={corr_delta:.4f}, p={p_delta:.4f} {'*' if p_delta < 0.05 else ''}")
    
    return results


def create_calibration_plot(iv_df, output_dir="."):
    """Create calibration plot comparing forecast vs learned model."""
    print("\n📊 Creating Calibration Plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: ROC Curve
    ax1 = axes[0]
    
    from sklearn.metrics import roc_curve
    
    y_true = iv_df['synthesis']
    
    if 'synthesis_probability' in iv_df.columns:
        fpr_forecast, tpr_forecast, _ = roc_curve(y_true, iv_df['synthesis_probability'])
        ax1.plot(fpr_forecast, tpr_forecast, label=f'Forecast (AUC={roc_auc_score(y_true, iv_df["synthesis_probability"]):.3f})', linewidth=2)
    
    if 'synthesis_prob_learned' in iv_df.columns:
        fpr_learned, tpr_learned, _ = roc_curve(y_true, iv_df['synthesis_prob_learned'])
        ax1.plot(fpr_learned, tpr_learned, label=f'Learned (AUC={roc_auc_score(y_true, iv_df["synthesis_prob_learned"]):.3f})', linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    ax2 = axes[1]
    
    from sklearn.metrics import precision_recall_curve
    
    if 'synthesis_probability' in iv_df.columns:
        precision_forecast, recall_forecast, _ = precision_recall_curve(y_true, iv_df['synthesis_probability'])
        ax2.plot(recall_forecast, precision_forecast, 
                label=f'Forecast (AUC={average_precision_score(y_true, iv_df["synthesis_probability"]):.3f})', linewidth=2)
    
    if 'synthesis_prob_learned' in iv_df.columns:
        precision_learned, recall_learned, _ = precision_recall_curve(y_true, iv_df['synthesis_prob_learned'])
        ax2.plot(recall_learned, precision_learned,
                label=f'Learned (AUC={average_precision_score(y_true, iv_df["synthesis_prob_learned"]):.3f})', linewidth=2)
    
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    cal_file = Path(output_dir) / "calibration_curves.png"
    plt.savefig(cal_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved calibration plots to: {cal_file}")
    plt.close()


def generate_calibration_report(iv_df, forecast_metrics, model_metrics, granger_results, output_file="calibration_report.txt"):
    """Generate calibration report."""
    print("\n📝 Generating Calibration Report...")
    
    report = []
    report.append("=" * 80)
    report.append("📊 BACKTEST & CALIBRATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Forecast evaluation
    report.append("=" * 80)
    report.append("🎯 SYNTHESIS FORECAST EVALUATION")
    report.append("=" * 80)
    report.append(f"Precision: {forecast_metrics['precision']:.4f}")
    report.append(f"Recall: {forecast_metrics['recall']:.4f}")
    report.append(f"ROC-AUC: {forecast_metrics['roc_auc']:.4f}")
    report.append(f"PR-AUC: {forecast_metrics['pr_auc']:.4f}")
    report.append(f"Brier Score: {forecast_metrics['brier']:.4f} (lower is better)")
    report.append("")
    
    # Learned model
    report.append("=" * 80)
    report.append("🔬 LEARNED MODEL PERFORMANCE")
    report.append("=" * 80)
    report.append(f"Precision: {model_metrics['precision']:.4f}")
    report.append(f"Recall: {model_metrics['recall']:.4f}")
    report.append(f"ROC-AUC: {model_metrics['roc_auc']:.4f}")
    report.append(f"PR-AUC: {model_metrics['pr_auc']:.4f}")
    report.append(f"Brier Score: {model_metrics['brier']:.4f}")
    report.append("")
    
    # Granger causality
    report.append("=" * 80)
    report.append("🔍 GRANGER CAUSALITY TESTS")
    report.append("=" * 80)
    for key, result in granger_results.items():
        sig = '*' if result['significant'] else ''
        report.append(f"{key}: r={result['correlation']:.4f}, p={result['p_value']:.4f}{sig}")
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
    print("📊 Step 5: Backtest + Calibrate")
    print("=" * 80)
    
    # Load data
    print("\n📥 Loading data...")
    iv_df = pd.read_csv(iv_file)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    
    print(f"  Loaded {len(iv_df)} days")
    
    # Prepare targets
    iv_df = prepare_targets(iv_df)
    
    # Evaluate existing forecast
    forecast_metrics = evaluate_synthesis_forecast(iv_df)
    
    # Train learned model
    model, scaler, features, model_metrics = train_synthesis_model(iv_df, use_xgboost=HAS_XGBOOST)
    
    # Granger causality tests
    granger_results = granger_causality_test(iv_df, max_lag=2)
    
    # Create visualizations
    try:
        create_calibration_plot(iv_df, output_dir)
    except Exception as e:
        print(f"\n⚠️  Error creating plots: {e}")
    
    # Save enhanced data
    enhanced_file = "iv_metrics_calibrated.csv"
    iv_df.to_csv(enhanced_file, index=False)
    print(f"\n✓ Saved calibrated metrics to: {enhanced_file}")
    
    # Generate report
    report_text = generate_calibration_report(iv_df, forecast_metrics, model_metrics, granger_results)
    
    print("\n" + "=" * 80)
    print("✅ Calibration complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {enhanced_file} - Calibrated metrics with learned predictions")
    print(f"  - calibration_curves.png - ROC and PR curves")
    print(f"  - calibration_report.txt - Detailed calibration report")


if __name__ == "__main__":
    main()

