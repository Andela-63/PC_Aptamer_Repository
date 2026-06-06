"""
ml_ensemble.py
==============
RF + Gradient Boosting ensemble with Leave-One-Out cross-validation,
bootstrapped 95% confidence intervals, and three-method feature importance.

Author  : AbuHuwaiz Al-Rashidi
ORCID   : 0000-0002-6615-0664
License : MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


def build_models(random_state: int = 42):
    """Return configured RF and GBR estimators."""
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=random_state
    )
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        random_state=random_state
    )
    return rf, gbr


def bootstrap_ci(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 metric_fn,
                 n_bootstrap: int = 2000,
                 ci: float = 0.95,
                 random_state: int = 42) -> tuple:
    """
    Compute bootstrapped confidence interval for a given metric.

    Returns
    -------
    (lower, upper) : float tuple
    """
    rng = np.random.default_rng(random_state)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yb, pb = y_true[idx], y_pred[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_vals.append(metric_fn(yb, pb))
    alpha = (1 - ci) / 2
    return (np.percentile(boot_vals, 100 * alpha),
            np.percentile(boot_vals, 100 * (1 - alpha)))


def run_loo_cv(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Run LOO-CV for both RF and GBR. Returns a dict with predictions,
    metrics, and 95% bootstrapped CIs.
    """
    rf, gbr = build_models(random_state)
    loo = LeaveOneOut()

    y_rf  = cross_val_predict(rf,  X, y, cv=loo)
    y_gbr = cross_val_predict(gbr, X, y, cv=loo)

    results = {}
    for name, y_pred in [('RF', y_rf), ('GBR', y_gbr)]:
        r2  = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        ci_r2  = bootstrap_ci(y, y_pred, r2_score)
        ci_mae = bootstrap_ci(y, y_pred, mean_absolute_error)
        results[name] = {
            'y_pred':  y_pred,
            'R2':      r2,
            'MAE':     mae,
            'CI_R2':   ci_r2,
            'CI_MAE':  ci_mae,
        }

    print("=" * 58)
    print("LOO-CV RESULTS WITH BOOTSTRAPPED 95% CI  (B = 2,000)")
    print("=" * 58)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  R²  = {res['R2']:.4f}  "
              f"(95% CI: {res['CI_R2'][0]:.4f} – {res['CI_R2'][1]:.4f})")
        print(f"  MAE = {res['MAE']:.4f}  "
              f"(95% CI: {res['CI_MAE'][0]:.4f} – {res['CI_MAE'][1]:.4f})")

    return results


def fit_ensemble(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Fit RF and GBR on the full dataset.

    Returns
    -------
    rf, gbr, scaler : fitted estimators and scaler
    """
    rf, gbr = build_models(random_state)
    rf.fit(X, y)
    gbr.fit(X, y)
    return rf, gbr


def compute_feature_importance(rf, gbr, X: np.ndarray, y: np.ndarray,
                                feature_names: list,
                                n_repeats: int = 100,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Three-method feature importance:
    1. MDI (mean decrease in impurity — RF only)
    2. Permutation importance (100 repeats, model-agnostic)
    3. Pearson and Spearman correlations with y

    Returns a DataFrame sorted by permutation importance (descending).
    """
    # Permutation importance
    pi = permutation_importance(rf, X, y,
                                n_repeats=n_repeats,
                                random_state=random_state)

    feat_df = pd.DataFrame({
        'feature':       feature_names,
        'MDI_importance':rf.feature_importances_,
        'perm_mean':     pi.importances_mean,
        'perm_std':      pi.importances_std,
        'pearson_r':     [stats.pearsonr(X[:, i], y)[0]
                          for i in range(X.shape[1])],
        'spearman_r':    [stats.spearmanr(X[:, i], y)[0]
                          for i in range(X.shape[1])],
    }).sort_values('perm_mean', ascending=False)

    print("\nTOP 15 FEATURES — permutation importance")
    print(feat_df.head(15)[['feature', 'perm_mean', 'perm_std',
                              'pearson_r', 'spearman_r']].to_string(index=False))
    return feat_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from feature_engineering import load_training_data, FEATURE_NAMES
    from sklearn.preprocessing import StandardScaler

    X_raw, y, meta = load_training_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("Training dataset shape:", X.shape)
    results = run_loo_cv(X, y)
    rf, gbr = fit_ensemble(X, y)
    feat_df = compute_feature_importance(rf, gbr, X, y, FEATURE_NAMES)
