#!/usr/bin/env python3
"""
Midterm: DPSyn-style Differentially Private Synthetic Data on Adult

This is an Adult-specific implementation inspired by the DPSyn pipeline:
1) discretize Adult into a fully categorical/discrete table
2) select many low-order marginals (1-way, 2-way, 3-way)
3) add Gaussian noise to those marginals under DP
4) enforce non-negativity / approximate consistency
5) synthesize records by iterative repair against the noisy marginals
6) evaluate downstream ML utility on the real test set

"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

# This block defines the column names and feature groups used throughout the
# experiment. The Adult dataset contains both numerical and categorical columns.
# Numerical columns are later discretized into bins, while categorical columns
# are encoded into integer IDs. The income column is used as the target/sensitive
# attribute for downstream classification and utility evaluation.
# =========================================================
# Adult dataset schema and experiment constants
# =========================================================
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

# This dataclass stores all information learned during preprocessing. It keeps
# the numeric bin edges, categorical encoding maps, inverse maps for converting
# synthetic data back to readable form, interval labels, feature names, and
# domain sizes. Keeping these artifacts ensures that real train, real test, and
# synthetic data use the same representation.
# =========================================================
# Preprocessing metadata container
# =========================================================

@dataclass
class PreprocessArtifacts:
    bin_edges: Dict[str, np.ndarray]
    category_maps: Dict[str, Dict[str, int]]
    inverse_category_maps: Dict[str, Dict[int, str]]
    interval_labels: Dict[str, Dict[int, str]]
    feature_cols: List[str]
    domain_sizes: Dict[str, int]

# This section loads the Adult dataset, cleans missing values, standardizes
# labels, and converts the mixed-type table into a fully discrete representation.
# Differentially private marginal-based synthesis requires every attribute to
# have a finite domain, so numerical columns are converted into bins and
# categorical columns are converted into integer codes.
# =========================================================
# Loading and preprocessing
# =========================================================

def load_adult_dataframe(path: str) -> pd.DataFrame:
    """
    Load the Adult dataset from CSV.

    Supports both raw UCI Adult format without column headers and CSV files
    that already contain headers. The function also normalizes whitespace,
    removes missing values marked as '?', and standardizes income labels such
    as '>50K.' to '>50K'.

    Args:
        path: Path to the Adult CSV file.

    Returns:
        Cleaned Adult dataframe with the expected Adult column names.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] == len(ADULT_COLUMNS):
        df.columns = ADULT_COLUMNS
    else:
        df = pd.read_csv(path)
        if df.shape[1] != len(ADULT_COLUMNS):
            raise ValueError(
                f"Unexpected Adult dataset shape/columns: {df.shape}, {list(df.columns)}"
            )
        df.columns = ADULT_COLUMNS

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].replace(
        {"<=50K.": "<=50K", ">50K.": ">50K"}
    )
    return df


def _safe_qcut_edges(series: pd.Series, q: int = 8) -> np.ndarray:
    """
    Create robust quantile-based bin edges for a numeric column.

    This is used to discretize continuous Adult attributes before DP synthetic
    generation. If the column has too many repeated values and quantile binning
    fails to produce enough unique edges, the function falls back to a simpler
    min/max bin.

    Args:
        series: Numeric pandas Series.
        q: Number of desired quantile bins.

    Returns:
        Array of bin edges.
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
    Convert numeric bin boundaries into a human-readable interval label.

    Example:
        left=25, right=37 -> '(25, 37]'

    Args:
        left: Left boundary of the interval.
        right: Right boundary of the interval.

    Returns:
        String representation of the interval.
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
    Learn all preprocessing metadata from the training data.

    Numeric columns are discretized into quantile bins. Categorical columns are
    encoded into integer IDs. This function stores the bin edges, category maps,
    inverse maps, interval labels, and domain sizes needed later for DP marginal
    computation and inverse transformation.

    Args:
        train_df: Real Adult training dataframe.
        n_bins: Number of bins for numeric attributes.

    Returns:
        PreprocessArtifacts object containing all preprocessing metadata.
    """
    bin_edges: Dict[str, np.ndarray] = {}
    category_maps: Dict[str, Dict[str, int]] = {}
    inverse_category_maps: Dict[str, Dict[int, str]] = {}
    interval_labels: Dict[str, Dict[int, str]] = {}
    domain_sizes: Dict[str, int] = {}

    for col in NUMERIC_COLS:
        edges = _safe_qcut_edges(train_df[col], q=n_bins)
        edges = np.unique(edges)
        if len(edges) < 2:
            v = float(train_df[col].iloc[0])
            edges = np.array([-np.inf, v, np.inf])
        bin_edges[col] = edges
        interval_labels[col] = {
            i: _make_interval_label(edges[i], edges[i + 1])
            for i in range(len(edges) - 1)
        }
        domain_sizes[col] = len(edges) - 1

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        vocab = sorted(train_df[col].astype(str).unique().tolist())
        if "<UNK>" not in vocab:
            vocab.append("<UNK>")
        mapping = {v: i for i, v in enumerate(vocab)}
        inv_mapping = {i: v for v, i in mapping.items()}
        category_maps[col] = mapping
        inverse_category_maps[col] = inv_mapping
        domain_sizes[col] = len(vocab)

    return PreprocessArtifacts(
        bin_edges=bin_edges,
        category_maps=category_maps,
        inverse_category_maps=inverse_category_maps,
        interval_labels=interval_labels,
        feature_cols=NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL],
        domain_sizes=domain_sizes,
    )


def transform_dataframe(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    """
    Transform a real Adult dataframe into a fully discrete integer-coded table.

    Numeric features are converted into bin IDs, and categorical features are
    converted into integer IDs using the mappings learned from the training set.
    This discrete format is required for marginal counting and DP synthetic data
    generation.

    Args:
        df: Adult dataframe to transform.
        artifacts: Preprocessing metadata learned from fit_preprocessor.

    Returns:
        Discrete integer-coded dataframe.
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
    Convert a discrete synthetic dataset back into human-readable form.

    Numeric bin IDs are mapped to interval labels, and categorical integer IDs
    are mapped back to category names. This output is easier to inspect and can
    be used as the released synthetic dataset.

    Args:
        synth_df: Integer-coded synthetic dataframe.
        artifacts: Preprocessing metadata.

    Returns:
        Human-readable synthetic dataframe.
    """
    out = pd.DataFrame(index=synth_df.index)

    for col in NUMERIC_COLS:
        out[col] = synth_df[col].astype(int).map(artifacts.interval_labels[col])

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        out[col] = synth_df[col].astype(int).map(artifacts.inverse_category_maps[col])

    return out

# This section converts the requested privacy budget, represented by epsilon and
# delta, into the Gaussian noise scale used for marginal measurements. Because
# many marginals are measured, the code uses a conservative composition strategy:
# more measured queries require more noise to satisfy the same overall privacy
# budget.
# =========================================================
# DP accounting
# =========================================================

def solve_rho_from_epsilon_delta(epsilon: float, delta: float) -> float:
    """
    Convert an (epsilon, delta)-DP target into a zCDP rho value.

    The code uses the relationship:
        epsilon = rho + 2 * sqrt(rho * log(1 / delta))

    This rho value is then used to calibrate Gaussian noise for marginal
    measurements.

    Args:
        epsilon: DP privacy budget. Smaller means stronger privacy.
        delta: DP failure probability.

    Returns:
        rho value for zCDP-style Gaussian noise calibration.
    """
    if not (epsilon > 0):
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    a = math.sqrt(math.log(1.0 / delta))
    t = -a + math.sqrt(a * a + epsilon)
    return max(t * t, 1e-12)


def gaussian_sigma_for_k_queries(epsilon: float, delta: float, k_queries: int) -> float:
    """
    Compute Gaussian noise scale for multiple marginal measurements.

    The total privacy budget is split across all measured marginals using a
    conservative zCDP-style composition rule. More queries require larger noise.

    Args:
        epsilon: Target epsilon privacy budget.
        delta: Target delta value.
        k_queries: Number of marginal queries to release.

    Returns:
        Standard deviation of Gaussian noise.
    """
    rho_total = solve_rho_from_epsilon_delta(epsilon, delta)
    sigma = math.sqrt(k_queries / (2.0 * rho_total))
    return sigma

# This section chooses which statistics of the real data will be measured with
# differential privacy. The workload includes all 1-way marginals, important
# 2-way marginals selected by mutual information, target-related 3-way marginals,
# and some random marginals for broader coverage. These noisy marginals become
# the statistical blueprint for generating synthetic data.
# =========================================================
# Marginal workload selection
# =========================================================

def select_workload_dpsyn(
    train_discrete: pd.DataFrame,
    top_pairs: int = 25,
    top_triples: int = 10,
    random_pairs: int = 15,
    random_triples: int = 8,
    seed: int = 42,
) -> List[Tuple[str, ...]]:
    """
    Select the set of marginals to measure privately.

    The workload includes all 1-way marginals, high-mutual-information 2-way
    marginals, target-related 3-way marginals, and some random marginals for
    broader data coverage. This follows the DPSyn idea of using many noisy
    low-dimensional marginals to reconstruct synthetic records.

    Args:
        train_discrete: Discrete training dataframe.
        top_pairs: Number of strongest 2-way marginals to include.
        top_triples: Number of strongest 3-way target-related marginals.
        random_pairs: Number of additional random 2-way marginals.
        random_triples: Number of additional random 3-way marginals.
        seed: Random seed for reproducibility.

    Returns:
        List of marginal projections, where each projection is a tuple of column names.
    """
    rng = random.Random(seed)
    cols = list(train_discrete.columns)

    workload: List[Tuple[str, ...]] = [(c,) for c in cols]

    pair_scores = []
    for a, b in combinations(cols, 2):
        mi = mutual_info_score(train_discrete[a], train_discrete[b])
        pair_scores.append(((a, b), float(mi)))
    pair_scores.sort(key=lambda x: x[1], reverse=True)

    workload.extend([pair for pair, _ in pair_scores[:top_pairs]])

    non_target = [c for c in cols if c != TARGET_COL]
    triple_scores = []
    for a, b in combinations(non_target, 2):
        score = (
            mutual_info_score(train_discrete[a], train_discrete[TARGET_COL]) +
            mutual_info_score(train_discrete[b], train_discrete[TARGET_COL]) +
            0.5 * mutual_info_score(train_discrete[a], train_discrete[b])
        )
        triple_scores.append(((a, b, TARGET_COL), float(score)))
    triple_scores.sort(key=lambda x: x[1], reverse=True)

    workload.extend([triple for triple, _ in triple_scores[:top_triples]])

    all_pairs = list(combinations(cols, 2))
    strong_pairs = set([pair for pair, _ in pair_scores[:top_pairs]])
    candidate_pairs = [p for p in all_pairs if p not in strong_pairs]
    rng.shuffle(candidate_pairs)
    workload.extend(candidate_pairs[:random_pairs])

    all_triples = list(combinations(cols, 3))
    strong_triples = set([triple for triple, _ in triple_scores[:top_triples]])
    candidate_triples = [t for t in all_triples if t not in strong_triples]
    rng.shuffle(candidate_triples)
    workload.extend(candidate_triples[:random_triples])

    deduped = []
    seen = set()
    for proj in workload:
        if proj not in seen:
            deduped.append(proj)
            seen.add(proj)
    return deduped

# This section computes contingency tables for selected marginals and adds
# calibrated Gaussian noise to them. For example, a marginal over education and
# income counts how many records fall into each education-income combination.
# Noise is added so that the contribution of any single person is protected.
# =========================================================
# Marginal counting and noisy measurement
# =========================================================

def marginal_counts(df: pd.DataFrame, attrs: Tuple[str, ...], domain_sizes: Dict[str, int]) -> np.ndarray:
    """
    Compute the contingency table for a selected marginal.

    For example, if attrs=('education', 'income'), this function counts how many
    records fall into each education-income combination.

    Args:
        df: Discrete dataframe.
        attrs: Columns included in the marginal.
        domain_sizes: Number of possible values for each column.

    Returns:
        N-dimensional numpy array of counts.
    """
    shape = [domain_sizes[a] for a in attrs]
    counts = np.zeros(shape, dtype=float)

    arr = df[list(attrs)].to_numpy(dtype=int)
    for row in arr:
        counts[tuple(row)] += 1.0
    return counts


def add_dp_noise_to_marginals(
    train_discrete: pd.DataFrame,
    workload: Sequence[Tuple[str, ...]],
    domain_sizes: Dict[str, int],
    epsilon: float,
    delta: float,
    random_state: int = 42,
) -> Dict[Tuple[str, ...], np.ndarray]:
    """
    Measure selected marginals with differential privacy.

    Each marginal count table is computed from the real data and Gaussian noise
    is added to protect individual records. Negative noisy counts are clipped to
    zero before later consistency repair.

    Args:
        train_discrete: Discrete real training data.
        workload: List of marginals to measure.
        domain_sizes: Domain size of each attribute.
        epsilon: DP privacy budget.
        delta: DP failure probability.
        random_state: Random seed.

    Returns:
        Dictionary mapping each marginal projection to its noisy count table.
    """
    rng = np.random.default_rng(random_state)
    sigma = gaussian_sigma_for_k_queries(epsilon, delta, len(workload))

    noisy = {}
    for attrs in workload:
        counts = marginal_counts(train_discrete, attrs, domain_sizes)
        noisy_counts = counts + rng.normal(loc=0.0, scale=sigma, size=counts.shape)
        noisy_counts = np.maximum(noisy_counts, 0.0)
        noisy[attrs] = noisy_counts
    return noisy


def normalize_counts_to_total(counts: np.ndarray, total_n: int) -> np.ndarray:
    """
    Normalize a noisy count table so that its total equals the dataset size.

    This step also ensures all counts are non-negative. If the table contains no
    positive mass after clipping, it is replaced with a uniform distribution.

    Args:
        counts: Noisy marginal count table.
        total_n: Target total number of records.

    Returns:
        Normalized non-negative count table.
    """
    counts = np.maximum(counts, 0.0)
    s = counts.sum()
    if s <= 0:
        flat = np.ones_like(counts, dtype=float)
        flat = flat / flat.sum()
        return flat * total_n
    return counts * (total_n / s)


def make_marginals_consistent(
    noisy_marginals: Dict[Tuple[str, ...], np.ndarray],
    total_n: int,
    rounds: int = 8,
) -> Dict[Tuple[str, ...], np.ndarray]:
    """
    Perform lightweight consistency repair across noisy marginals.

    DP noise can make related marginals inconsistent. For example, the sum of a
    2-way education-income marginal may not match the 1-way income marginal.
    This function repeatedly normalizes marginals and partially aligns lower-order
    marginals with projections from higher-order marginals.

    Args:
        noisy_marginals: Dictionary of noisy marginal tables.
        total_n: Target number of synthetic records.
        rounds: Number of consistency repair iterations.

    Returns:
        Approximately consistent marginal tables.
    """
    repaired = {
        attrs: normalize_counts_to_total(arr.copy(), total_n)
        for attrs, arr in noisy_marginals.items()
    }

    by_len = defaultdict(list)
    for attrs in repaired:
        by_len[len(attrs)].append(attrs)

    for _ in range(rounds):
        # make every marginal sum to total_n
        for attrs in repaired:
            repaired[attrs] = normalize_counts_to_total(repaired[attrs], total_n)

        # project longer marginals onto shorter ones and average
        updates = defaultdict(list)

        for attrs_long, arr_long in repaired.items():
            if len(attrs_long) <= 1:
                continue

            for r in range(1, len(attrs_long)):
                for sub_attrs in combinations(attrs_long, r):
                    axes_to_sum = tuple(i for i, a in enumerate(attrs_long) if a not in sub_attrs)
                    projected = arr_long.sum(axis=axes_to_sum)
                    updates[sub_attrs].append(projected)

        for attrs_short, projected_list in updates.items():
            if attrs_short in repaired and projected_list:
                avg = np.mean(np.stack(projected_list, axis=0), axis=0)
                repaired[attrs_short] = 0.5 * repaired[attrs_short] + 0.5 * avg
                repaired[attrs_short] = np.maximum(repaired[attrs_short], 0.0)

    for attrs in repaired:
        repaired[attrs] = normalize_counts_to_total(repaired[attrs], total_n)

    return repaired


# =========================================================
# Synthesis
# =========================================================

def sample_from_1way_marginals(
    marginals: Dict[Tuple[str, ...], np.ndarray],
    feature_cols: List[str],
    n_records: int,
    domain_sizes: Dict[str, int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create an initial synthetic dataset using 1-way marginals.

    Each column is sampled independently according to its noisy repaired 1-way
    distribution. This initial table captures single-column distributions but
    not yet relationships between columns.

    Args:
        marginals: Repaired marginal tables.
        feature_cols: Columns to generate.
        n_records: Number of synthetic records.
        domain_sizes: Domain size of each attribute.
        random_state: Random seed.

    Returns:
        Initial discrete synthetic dataframe.
    """
    rng = np.random.default_rng(random_state)
    synth = pd.DataFrame(index=np.arange(n_records))

    for col in feature_cols:
        counts = marginals[(col,)].astype(float)
        probs = counts / max(counts.sum(), 1e-12)
        synth[col] = rng.choice(np.arange(domain_sizes[col]), size=n_records, p=probs)

    return synth


def tuple_counts(df: pd.DataFrame, attrs: Tuple[str, ...]) -> Dict[Tuple[int, ...], int]:
    """
    Count observed value combinations for selected columns in a dataframe.

    This is used during synthetic record repair to compare the current synthetic
    marginal with the target noisy marginal.

    Args:
        df: Discrete synthetic dataframe.
        attrs: Columns included in the marginal.

    Returns:
        Dictionary mapping value tuples to counts.
    """
    vals = [tuple(x) for x in df[list(attrs)].to_numpy(dtype=int)]
    out = defaultdict(int)
    for v in vals:
        out[v] += 1
    return out


def target_tuple_counts(arr: np.ndarray) -> Dict[Tuple[int, ...], float]:
    """
    Convert a marginal count array into a dictionary indexed by value tuples.

    This format is easier to compare against observed tuple counts in the current
    synthetic dataset during the repair process.

    Args:
        arr: N-dimensional marginal count array.

    Returns:
        Dictionary mapping tuple indices to target counts.
    """
    out = {}
    for idx in np.ndindex(arr.shape):
        out[idx] = float(arr[idx])
    return out


def repair_synthetic_data_to_marginals(
    synth_df: pd.DataFrame,
    target_marginals: Dict[Tuple[str, ...], np.ndarray],
    domain_sizes: Dict[str, int],
    n_passes: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Iteratively repair synthetic records to better match noisy target marginals.

    The algorithm finds overrepresented value combinations in the synthetic data
    and changes some records toward underrepresented combinations. This is a
    simplified DPSyn-style reconstruction step.

    Args:
        synth_df: Initial discrete synthetic dataframe.
        target_marginals: Repaired noisy marginals to match.
        domain_sizes: Domain size of each attribute.
        n_passes: Number of repair passes over the workload.
        random_state: Random seed.

    Returns:
        Repaired discrete synthetic dataframe.
    """
    rng = np.random.default_rng(random_state)
    synth = synth_df.copy()

    workload = sorted(target_marginals.keys(), key=len, reverse=True)

    for _ in range(n_passes):
        rng.shuffle(workload)

        for attrs in workload:
            current = tuple_counts(synth, attrs)
            target = target_tuple_counts(target_marginals[attrs])

            deficits = []
            surpluses = []

            for tup, tval in target.items():
                cval = current.get(tup, 0)
                diff = tval - cval
                if diff > 0.5:
                    deficits.append((tup, diff))
                elif diff < -0.5:
                    surpluses.append((tup, -diff))

            if not deficits or not surpluses:
                continue

            deficits.sort(key=lambda x: x[1], reverse=True)
            surpluses.sort(key=lambda x: x[1], reverse=True)

            max_changes = min(2000, len(synth) // 10 + 1)

            changes = 0
            for surplus_tup, surplus_amt in surpluses:
                if changes >= max_changes:
                    break

                mask = np.ones(len(synth), dtype=bool)
                for a, v in zip(attrs, surplus_tup):
                    mask &= (synth[a].to_numpy() == v)

                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue

                rng.shuffle(idxs)
                movable = idxs[: int(min(len(idxs), math.ceil(surplus_amt)))]

                for row_idx in movable:
                    if changes >= max_changes or not deficits:
                        break

                    deficit_tup, deficit_amt = deficits[0]

                    for a, v in zip(attrs, deficit_tup):
                        synth.at[row_idx, a] = int(v)

                    deficit_amt -= 1.0
                    changes += 1

                    if deficit_amt <= 0.5:
                        deficits.pop(0)
                        if not deficits:
                            break
                    else:
                        deficits[0] = (deficit_tup, deficit_amt)

    for col in synth.columns:
        synth[col] = synth[col].clip(0, domain_sizes[col] - 1).astype(int)

    # Force Adult income target to binary only
    if "income" in synth.columns:
        synth["income"] = synth["income"].apply(lambda x: 1 if x == 1 else 0).astype(int)

    return synth


# =========================================================
# Evaluation
# =========================================================

def build_tabular_models(random_state: int = 42) -> Dict[str, object]:
    """
    Define downstream machine learning models for utility evaluation.

    These models are trained on synthetic data and tested on the real test set.
    The same set of models can be compared with HW1 results.

    Args:
        random_state: Random seed for reproducible model training.

    Returns:
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

    Models are trained on the provided training dataframe and evaluated on the
    real held-out test dataframe. All features are one-hot encoded because the
    data is discrete/categorical.

    Args:
        train_df: Training dataframe, usually synthetic data.
        test_df: Real test dataframe.
        random_state: Random seed.

    Returns:
        Dataframe containing accuracy, misclassification rate, precision, recall,
        and AUC for each model.
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
        elif hasattr(pipe, "decision_function"):
            raw = np.asarray(pipe.decision_function(X_test))
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

# This section runs the full DPSyn-style experiment. It loads the Adult dataset,
# creates train/test splits, preprocesses the data, selects marginals, generates
# DP synthetic datasets for several epsilon values, evaluates ML utility, and
# saves all output CSV files. This is the entry point executed when the script is
# run from the command line.
# =========================================================
# Main experiment pipeline
# =========================================================

def save_json(obj: dict, path: Path) -> None:
    """
    Save experiment metadata to a JSON file.

    This is used to store the selected workload and configuration parameters so
    the experiment can be reproduced later.

    Args:
        obj: Python dictionary to save.
        path: Output JSON file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    """
    Run the full DPSyn-style Adult experiment.

    Steps:
    1. Load and clean Adult data.
    2. Split into train/test sets.
    3. Discretize and encode the data.
    4. Evaluate real-data baseline.
    5. Select marginal workload.
    6. For each epsilon:
       - add DP noise to marginals,
       - repair marginal consistency,
       - generate synthetic data,
       - evaluate downstream ML utility.
    7. Save all synthetic datasets and metric files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="hw3_dpsyn_outputs")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=8)
    parser.add_argument("--top_pairs", type=int, default=25)
    parser.add_argument("--top_triples", type=int, default=10)
    parser.add_argument("--random_pairs", type=int, default=15)
    parser.add_argument("--random_triples", type=int, default=8)
    parser.add_argument("--consistency_rounds", type=int, default=8)
    parser.add_argument("--repair_passes", type=int, default=6)
    parser.add_argument("--n_records", type=int, default=0,
                        help="If 0, use same size as real train split")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataframe(args.adult_path)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[TARGET_COL],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    artifacts = fit_preprocessor(train_df, n_bins=args.n_bins)
    train_discrete = transform_dataframe(train_df, artifacts)
    test_discrete = transform_dataframe(test_df, artifacts)

    train_discrete.to_csv(outdir / "adult_train_discrete.csv", index=False)
    test_discrete.to_csv(outdir / "adult_test_discrete.csv", index=False)

    baseline_metrics = evaluate_downstream_models(
        train_df=train_discrete,
        test_df=test_discrete,
        random_state=args.random_state,
    )
    baseline_metrics.insert(0, "setting", "original_real")
    baseline_metrics.insert(1, "epsilon", np.nan)
    baseline_metrics.to_csv(outdir / "metrics_original_real.csv", index=False)

    workload = select_workload_dpsyn(
        train_discrete=train_discrete,
        top_pairs=args.top_pairs,
        top_triples=args.top_triples,
        random_pairs=args.random_pairs,
        random_triples=args.random_triples,
        seed=args.random_state,
    )

    save_json(
        {
            "workload": [list(w) for w in workload],
            "delta": args.delta,
            "n_bins": args.n_bins,
            "top_pairs": args.top_pairs,
            "top_triples": args.top_triples,
            "random_pairs": args.random_pairs,
            "random_triples": args.random_triples,
            "consistency_rounds": args.consistency_rounds,
            "repair_passes": args.repair_passes,
        },
        outdir / "workload.json",
    )

    all_results = [baseline_metrics]
    n_records = args.n_records if args.n_records > 0 else len(train_discrete)

    for eps in args.epsilons:
        print("\n" + "=" * 80)
        print(f"DPSyn-style generation for epsilon={eps}")
        print("=" * 80)

        noisy_marginals = add_dp_noise_to_marginals(
            train_discrete=train_discrete,
            workload=workload,
            domain_sizes=artifacts.domain_sizes,
            epsilon=eps,
            delta=args.delta,
            random_state=args.random_state,
        )

        consistent_marginals = make_marginals_consistent(
            noisy_marginals=noisy_marginals,
            total_n=n_records,
            rounds=args.consistency_rounds,
        )

        init_synth = sample_from_1way_marginals(
            marginals=consistent_marginals,
            feature_cols=artifacts.feature_cols,
            n_records=n_records,
            domain_sizes=artifacts.domain_sizes,
            random_state=args.random_state,
        )

        synth_discrete = repair_synthetic_data_to_marginals(
            synth_df=init_synth,
            target_marginals=consistent_marginals,
            domain_sizes=artifacts.domain_sizes,
            n_passes=args.repair_passes,
            random_state=args.random_state,
        )

        synth_discrete.to_csv(outdir / f"adult_synth_discrete_eps_{eps}.csv", index=False)

        synth_human = inverse_transform_synthetic(synth_discrete, artifacts)
        synth_human.to_csv(outdir / f"adult_synth_human_eps_{eps}.csv", index=False)

        synth_metrics = evaluate_downstream_models(
            train_df=synth_discrete,
            test_df=test_discrete,
            random_state=args.random_state,
        )
        synth_metrics.insert(0, "setting", "dpsyn_synthetic")
        synth_metrics.insert(1, "epsilon", eps)
        synth_metrics.to_csv(outdir / f"metrics_dpsyn_eps_{eps}.csv", index=False)

        all_results.append(synth_metrics)

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv(outdir / "metrics_all.csv", index=False)

    print("\nSaved outputs to:", outdir.resolve())
    print(final_results)


if __name__ == "__main__":
    main()
