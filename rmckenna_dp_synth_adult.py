#!/usr/bin/env python3
"""
HW3: Differentially Private Synthetic Data on Adult
Adapted from the rmckenna / Private-PGM style workflow:

1) preprocess Adult into a fully discrete tabular dataset
2) select informative 1-way / 2-way / 3-way marginals
3) measure them with Gaussian noise under DP
4) fit a graphical model with Private-PGM
5) generate a synthetic training dataset
6) evaluate downstream ML utility on the real test set
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.cluster import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

try:
    from tmlt.private_pgm import Dataset, Domain, FactoredInference
except ImportError as exc:
    raise SystemExit(
        "Please install Private-PGM first:\n"
        "  pip install tmlt.private_pgm\n"
    ) from exc

# Defines the Adult dataset schema and separates numerical, categorical,
# and target attributes used throughout the experiment.
# ----------------------------
# Adult schema
# ----------------------------

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_COL = "income"


@dataclass
class PreprocessArtifacts:
    bin_edges: Dict[str, np.ndarray]
    category_maps: Dict[str, Dict[str, int]]
    inverse_category_maps: Dict[str, Dict[int, str]]
    interval_labels: Dict[str, Dict[int, str]]
    feature_cols: List[str]

# Converts the raw Adult dataset into the discrete finite-domain format
# required by marginal-based DP synthetic data generation.
# ----------------------------
# Loading and cleaning
# ----------------------------

def load_adult_dataframe(path: str) -> pd.DataFrame:
    """
    Load the Adult dataset from CSV and standardize its format.

    This function supports both raw UCI Adult files without headers and CSV files
    that already contain column names. It also removes records with missing values,
    strips whitespace from string columns, and normalizes income labels such as
    '>50K.' into '>50K'.

    Returns
    -------
    pd.DataFrame
        Cleaned Adult dataset with standardized column names.
    """

    df = pd.read_csv(path, header=None)
    if df.shape[1] == len(ADULT_COLUMNS):
        df.columns = ADULT_COLUMNS
    else:
        # Try again with header row
        df = pd.read_csv(path)
        if list(df.columns) != ADULT_COLUMNS:
            # If header names differ but count matches, rename to expected schema
            if df.shape[1] == len(ADULT_COLUMNS):
                df.columns = ADULT_COLUMNS
            else:
                raise ValueError(
                    f"Unexpected Adult dataset shape/columns: {df.shape}, {list(df.columns)}"
                )

    # Strip whitespace and normalize missing values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df = df.replace("?", np.nan).dropna().reset_index(drop=True)

    # Normalize target labels
    df[TARGET_COL] = df[TARGET_COL].replace(
        {
            "<=50K.": "<=50K",
            ">50K.": ">50K",
        }
    )

    return df

# Converts continuous and categorical Adult features into integer-coded
# finite domains required by Private-PGM.
# ----------------------------
# Discretization / encoding
# ----------------------------

def _safe_qcut_edges(series: pd.Series, q: int = 8) -> np.ndarray:
    """
    Compute robust quantile-based bin edges for numerical discretization.

    Differentially private marginal methods require finite discrete domains.
    Therefore, continuous numerical columns are converted into bins. This helper
    uses quantiles when possible and falls back to simple min/max edges when the
    column has too many duplicate values.

    Returns
    -------
    np.ndarray
        Monotonic bin edges used by pd.cut().
    """
    quantiles = np.linspace(0, 1, q + 1)
    edges = np.unique(series.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        mn, mx = series.min(), series.max()
        if mn == mx:
            edges = np.array([mn - 0.5, mx + 0.5], dtype=float)
        else:
            edges = np.array([mn, mx], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    return edges


def _make_interval_label(left: float, right: float) -> str:
    """
    Convert two bin boundaries into a readable interval label.

    This is used only for exporting human-readable synthetic data. Internally,
    the algorithm uses integer bin IDs, but the final CSV is easier to inspect
    when numerical bins are shown as intervals.

    Returns
    -------
    str
        Interval string such as '(25, 37]'.
    """
    def fmt(x: float) -> str:
        if np.isinf(x):
            return "inf" if x > 0 else "-inf"
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.2f}"

    return f"({fmt(left)}, {fmt(right)}]"


def fit_preprocessor(train_df: pd.DataFrame, n_bins: int = 8) -> PreprocessArtifacts:
    """
    Fit all preprocessing rules using only the training data.

    Numerical columns are discretized into quantile bins, and categorical columns
    are mapped to integer IDs. These mappings define the finite domain required
    by Private-PGM. The same preprocessing artifacts are later reused for the
    real test set and synthetic data.

    Returns
    -------
    PreprocessArtifacts
        Object containing bin edges, category mappings, inverse mappings, and
        readable interval labels.
    """
    bin_edges: Dict[str, np.ndarray] = {}
    category_maps: Dict[str, Dict[str, int]] = {}
    inverse_category_maps: Dict[str, Dict[int, str]] = {}
    interval_labels: Dict[str, Dict[int, str]] = {}

    # Numeric -> discrete bins
    for col in NUMERIC_COLS:
        edges = _safe_qcut_edges(train_df[col], q=n_bins)
        # pd.cut needs monotonic increasing edges
        edges = np.unique(edges)
        if len(edges) < 2:
            v = float(train_df[col].iloc[0])
            edges = np.array([-np.inf, v, np.inf])
        bin_edges[col] = edges

        labels = {}
        for i in range(len(edges) - 1):
            labels[i] = _make_interval_label(edges[i], edges[i + 1])
        interval_labels[col] = labels

    # Categorical -> integer encoding using train vocabulary
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        vocab = sorted(train_df[col].astype(str).unique().tolist())
        if "<UNK>" not in vocab:
            vocab.append("<UNK>")
        mapping = {v: i for i, v in enumerate(vocab)}
        inv_mapping = {i: v for v, i in mapping.items()}
        category_maps[col] = mapping
        inverse_category_maps[col] = inv_mapping

    return PreprocessArtifacts(
        bin_edges=bin_edges,
        category_maps=category_maps,
        inverse_category_maps=inverse_category_maps,
        interval_labels=interval_labels,
        feature_cols=NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL],
    )


def transform_dataframe(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    """
    Transform a raw Adult dataframe into a fully discrete integer dataframe.

    Numerical attributes are replaced by bin IDs, while categorical attributes
    are replaced by integer category IDs. This representation is required for
    computing marginals and fitting the graphical model.

    Returns
    -------
    pd.DataFrame
        Discrete version of the input dataframe.
    """
    out = pd.DataFrame(index=df.index)

    for col in NUMERIC_COLS:
        edges = artifacts.bin_edges[col]
        binned = pd.cut(
            df[col].astype(float),
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        )
        # In case of any edge-case NA after cut, assign to nearest valid bucket 0
        out[col] = binned.fillna(0).astype(int)

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        mapping = artifacts.category_maps[col]
        out[col] = df[col].astype(str).apply(lambda x: mapping.get(x, mapping["<UNK>"])).astype(int)

    return out


def inverse_transform_synthetic(
    synth_df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
) -> pd.DataFrame:
    """
    Convert synthetic integer-coded data back into human-readable form.

    Numerical bin IDs are converted into interval labels, and categorical IDs are
    converted back into category names. This output is useful for inspection and
    reporting, while model evaluation still uses the discrete version.

    Returns
    -------
    pd.DataFrame
        Human-readable synthetic Adult dataset.
    """
    out = pd.DataFrame(index=synth_df.index)

    for col in NUMERIC_COLS:
        labels = artifacts.interval_labels[col]
        out[col] = synth_df[col].astype(int).map(labels)

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        inv = artifacts.inverse_category_maps[col]
        out[col] = synth_df[col].astype(int).map(inv)

    return out

# Selects which statistical relationships should be preserved privately.
# Better workload selection usually improves synthetic data utility.
# ----------------------------
# Workload selection
# ----------------------------

def select_workload(
    train_discrete: pd.DataFrame,
    top_pairs: int = 20,
    top_triples: int = 8,
) -> List[Tuple[str, ...]]:
    """
    Select the set of marginals to privately measure.

    This function follows the R-McKenna / Private-PGM style idea: include all
    1-way marginals, select informative 2-way marginals using mutual information,
    and select several 3-way marginals involving the target variable. These
    marginals capture important statistical relationships used to generate
    useful synthetic data.

    Returns
    -------
    List[Tuple[str, ...]]
        List of marginal projections, such as ('education', 'income').
    """
    cols = list(train_discrete.columns)

    # 1-way marginals
    workload: List[Tuple[str, ...]] = [(c,) for c in cols]

    # Pairwise MI ranking
    pair_scores = []
    for a, b in combinations(cols, 2):
        mi = mutual_info_score(train_discrete[a], train_discrete[b])
        pair_scores.append(((a, b), float(mi)))

    pair_scores.sort(key=lambda x: x[1], reverse=True)
    selected_pairs = [pair for pair, _ in pair_scores[:top_pairs]]
    workload.extend(selected_pairs)

    # 3-way marginals centered on target column
    target_pairs = []
    non_target_cols = [c for c in cols if c != TARGET_COL]
    for a, b in combinations(non_target_cols, 2):
        score = (
            mutual_info_score(train_discrete[a], train_discrete[TARGET_COL]) +
            mutual_info_score(train_discrete[b], train_discrete[TARGET_COL]) +
            0.5 * mutual_info_score(train_discrete[a], train_discrete[b])
        )
        target_pairs.append(((a, b, TARGET_COL), float(score)))

    target_pairs.sort(key=lambda x: x[1], reverse=True)
    selected_triples = [triple for triple, _ in target_pairs[:top_triples]]
    workload.extend(selected_triples)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for proj in workload:
        if proj not in seen:
            deduped.append(proj)
            seen.add(proj)

    return deduped

# Calibrates Gaussian noise so that the marginal measurements satisfy
# approximate differential privacy.
# ----------------------------
# DP calibration
# ----------------------------

def solve_rho_from_epsilon_delta(epsilon: float, delta: float) -> float:
    """
    Convert an (epsilon, delta)-DP budget into a zCDP rho value.

    The Gaussian mechanism is often analyzed using zero-concentrated DP.
    This helper solves the standard conversion equation:

        epsilon = rho + 2 * sqrt(rho * log(1 / delta))

    Returns
    -------
    float
        Total rho privacy budget corresponding to the given epsilon and delta.
    """
    if not (epsilon > 0):
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    a = math.sqrt(math.log(1.0 / delta))
    t = -a + math.sqrt(a * a + epsilon)
    rho = max(t * t, 1e-12)
    return rho


def gaussian_sigma_for_k_queries(epsilon: float, delta: float, k_queries: int) -> float:
    """
    Compute the Gaussian noise scale for measuring multiple marginals.

    Each marginal measurement has sensitivity 1. Because multiple marginals are
    measured, the total privacy budget is divided across all measurements using
    a conservative zCDP composition rule.

    Returns
    -------
    float
        Standard deviation of Gaussian noise added to each marginal count.
    """
    rho_total = solve_rho_from_epsilon_delta(epsilon, delta)
    sigma = math.sqrt(k_queries / (2.0 * rho_total))
    return sigma

# Uses Private-PGM to estimate a global distribution from noisy marginals
# and generate synthetic records from that distribution.
# ----------------------------
# PGM fitting
# ----------------------------

def build_domain(train_discrete: pd.DataFrame) -> Domain:
    """
    Build the Private-PGM domain object from the discrete training data.

    The domain specifies every attribute name and the number of possible values
    for each attribute. Private-PGM uses this domain to understand the shape of
    the full data distribution and each marginal table.

    Returns
    -------
    Domain
        Private-PGM domain object.
    """
    attrs = list(train_discrete.columns)
    shape = [int(train_discrete[c].max()) + 1 for c in attrs]
    return Domain(attrs, shape)


def build_measurements(
    train_discrete: pd.DataFrame,
    workload: Sequence[Tuple[str, ...]],
    sigma: float,
) -> List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]:
    """
    Privately measure the selected marginals.

    For each marginal in the workload, this function computes the true histogram,
    adds Gaussian noise for differential privacy, and stores the result in the
    measurement format expected by Private-PGM.

    Returns
    -------
    list
        Measurement tuples of the form (Q, noisy_counts, sigma, projection).
    """
    domain = build_domain(train_discrete)
    data = Dataset(train_discrete, domain)

    measurements = []
    for proj in workload:
        hist = data.project(list(proj)).datavector()
        noisy = hist + np.random.normal(loc=0.0, scale=sigma, size=hist.size)
        Q = sparse.eye(hist.size, format="csr")
        measurements.append((Q, noisy, sigma, tuple(proj)))
    return measurements


def fit_private_pgm(
    train_discrete: pd.DataFrame,
    epsilon: float,
    delta: float,
    workload: Sequence[Tuple[str, ...]],
    pgm_iters: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit a differentially private graphical model and sample synthetic data.

    The function first calibrates Gaussian noise according to epsilon and delta,
    then measures selected marginals privately. Private-PGM estimates a global
    data distribution that approximately matches those noisy measurements.
    Finally, synthetic records are sampled from the estimated distribution.

    Returns
    -------
    pd.DataFrame
        Discrete synthetic Adult dataset with the same columns as the training data.
    """
    rng = np.random.default_rng(random_state)
    np.random.seed(random_state)

    domain = build_domain(train_discrete)
    sigma = gaussian_sigma_for_k_queries(epsilon, delta, len(workload))
    measurements = build_measurements(train_discrete, workload, sigma)

    engine = FactoredInference(domain, log=True, iters=pgm_iters)
    model = engine.estimate(measurements, engine="MD")

    n_rows = len(train_discrete)

    # Different Private-PGM versions may expose synthetic_data slightly differently.
    synth_dataset = None
    if hasattr(model, "synthetic_data"):
        try:
            synth_dataset = model.synthetic_data(rows=n_rows)
        except TypeError:
            synth_dataset = model.synthetic_data()

    if synth_dataset is None:
        raise RuntimeError(
            "Could not sample synthetic data from the fitted graphical model. "
            "Check your installed tmlt.private_pgm version."
        )

    synth_df = synth_dataset.df.copy()

    # Match requested number of rows if needed
    if len(synth_df) != n_rows:
        if len(synth_df) > n_rows:
            synth_df = synth_df.sample(n_rows, random_state=random_state).reset_index(drop=True)
        else:
            synth_df = synth_df.sample(
                n_rows,
                replace=True,
                random_state=random_state,
            ).reset_index(drop=True)

    # Ensure integer typing and valid domain bounds
    for col in train_discrete.columns:
        maxv = int(train_discrete[col].max())
        synth_df[col] = synth_df[col].round().clip(0, maxv).astype(int)

    return synth_df

# Trains downstream classifiers to evaluate how useful the synthetic data is
# for machine learning tasks.
# ----------------------------
# ML evaluation
# ----------------------------

def build_tabular_models(random_state: int = 42) -> Dict[str, object]:
    """
    Create downstream machine learning models for utility evaluation.

    These models are trained on either real or synthetic data and evaluated on
    the real test set. Using several model types helps measure whether the
    synthetic data preserves useful predictive patterns.

    Returns
    -------
    dict
        Dictionary mapping model names to scikit-learn model objects.
    """
    return {
        "LogisticRegression": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=random_state,
        ),
    }


def evaluate_downstream_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate synthetic data utility using downstream classification.

    A model is trained on the provided training data and tested on the real test
    data. If the synthetic data is useful, models trained on it should still
    achieve reasonable accuracy, precision, recall, and AUC on real data.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics for each downstream model.
    """
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("all_cat", OneHotEncoder(handle_unknown="ignore"), feature_cols),
        ]
    )

    rows = []
    models = build_tabular_models(random_state=random_state)

    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", model),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # fallback for SVM / others with decision function
            if hasattr(pipe, "decision_function"):
                raw = pipe.decision_function(X_test)
                raw = np.asarray(raw)
                proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
            else:
                proba = pred.astype(float)

        acc = accuracy_score(y_test, pred)
        mis = 1.0 - acc
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = np.nan

        rows.append(
            {
                "model": model_name,
                "accuracy": round(float(acc), 4),
                "misclassification_rate": round(float(mis), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "auc": round(float(auc), 4) if not np.isnan(auc) else np.nan,
            }
        )

    return pd.DataFrame(rows)

# Main experiment driver: runs preprocessing, synthesis, evaluation,
# and output saving for all epsilon values.
# ----------------------------
# Main experiment
# ----------------------------

def save_json(obj: dict, path: Path) -> None:
    """
    Save experiment metadata as a JSON file.

    This is used to store reproducibility information such as selected marginals,
    privacy parameters, and preprocessing settings.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    """
    Run the complete R-McKenna-style DP synthetic data experiment.

    The full pipeline is:
    1. Load and clean Adult data
    2. Split into train/test sets
    3. Discretize features
    4. Select informative marginals
    5. Generate DP synthetic data for each epsilon
    6. Train ML models on synthetic data
    7. Evaluate on the real test set
    8. Save synthetic datasets and metric CSV files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="hw3_outputs")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=8)
    parser.add_argument("--top_pairs", type=int, default=20)
    parser.add_argument("--top_triples", type=int, default=8)
    parser.add_argument("--pgm_iters", type=int, default=5000)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and split real data
    df = load_adult_dataframe(args.adult_path)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[TARGET_COL],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Fit preprocessing only on real train
    artifacts = fit_preprocessor(train_df, n_bins=args.n_bins)
    train_discrete = transform_dataframe(train_df, artifacts)
    test_discrete = transform_dataframe(test_df, artifacts)

    # Save transformed real data for reproducibility
    train_discrete.to_csv(outdir / "adult_train_discrete.csv", index=False)
    test_discrete.to_csv(outdir / "adult_test_discrete.csv", index=False)

    # Baseline on original real discrete train/test
    baseline_metrics = evaluate_downstream_models(
        train_df=train_discrete,
        test_df=test_discrete,
        random_state=args.random_state,
    )
    baseline_metrics.insert(0, "setting", "original_real")
    baseline_metrics.insert(1, "epsilon", np.nan)
    baseline_metrics.to_csv(outdir / "metrics_original_real.csv", index=False)

    # Select workload once from real train
    workload = select_workload(
        train_discrete=train_discrete,
        top_pairs=args.top_pairs,
        top_triples=args.top_triples,
    )

    save_json(
        {
            "workload": [list(w) for w in workload],
            "delta": args.delta,
            "top_pairs": args.top_pairs,
            "top_triples": args.top_triples,
            "n_bins": args.n_bins,
        },
        outdir / "workload.json",
    )

    all_results = [baseline_metrics]

    for eps in args.epsilons:
        print("\n" + "=" * 80)
        print(f"Generating DP synthetic data for epsilon={eps}")
        print("=" * 80)

        synth_discrete = fit_private_pgm(
            train_discrete=train_discrete,
            epsilon=eps,
            delta=args.delta,
            workload=workload,
            pgm_iters=args.pgm_iters,
            random_state=args.random_state,
        )

        # Save discrete synthetic train
        synth_discrete_path = outdir / f"adult_synth_discrete_eps_{eps}.csv"
        synth_discrete.to_csv(synth_discrete_path, index=False)

        # Save human-readable synthetic train
        synth_human = inverse_transform_synthetic(synth_discrete, artifacts)
        synth_human_path = outdir / f"adult_synth_human_eps_{eps}.csv"
        synth_human.to_csv(synth_human_path, index=False)

        # Evaluate by training on synthetic train and testing on real test
        synth_metrics = evaluate_downstream_models(
            train_df=synth_discrete,
            test_df=test_discrete,
            random_state=args.random_state,
        )
        synth_metrics.insert(0, "setting", "dp_synthetic")
        synth_metrics.insert(1, "epsilon", eps)
        synth_metrics.to_csv(outdir / f"metrics_dp_synth_eps_{eps}.csv", index=False)

        all_results.append(synth_metrics)

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv(outdir / "metrics_all.csv", index=False)

    print("\nSaved outputs to:", outdir.resolve())
    print(final_results)


if __name__ == "__main__":
    main()
