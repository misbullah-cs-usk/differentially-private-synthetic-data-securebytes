#!/usr/bin/env python3
"""
Adult DP-SGD with MLP

This script applies the HW2-style Differentially Private Stochastic Gradient
Descent (DP-SGD) training method to the Adult Income dataset.

Purpose:
  - Provide a direct Adult-dataset comparison against HW1 and HW3.
  - HW1 applies privacy during data release using k-anonymity/generalization.
  - HW3 applies privacy during data release using DP synthetic data.
  - This script applies privacy during model training using DP-SGD.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy_statement,
)

# =========================================================
# Adult dataset schema
# =========================================================
# The UCI Adult dataset may be loaded without column names.
# These names are assigned manually when the raw CSV has no header.
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

# Numerical features are standardized before training.
NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]

# Categorical features are converted to one-hot encoded vectors.
CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# Binary classification target.
TARGET_COL = "income"

# =========================================================
# Data loading
# =========================================================
def load_adult_dataframe(path: str) -> pd.DataFrame:
    """
    Load and clean the Adult dataset.

    This function supports two common formats:
      1. Raw UCI Adult file without headers.
      2. CSV file with headers.

    Cleaning steps:
      - Assign standard Adult column names.
      - Strip whitespace from categorical/string columns.
      - Replace '?' missing values with NaN.
      - Drop rows containing missing values.
      - Normalize income labels such as '<=50K.' and '>50K.'.
      - Create a binary target column:
          income_binary = 1 if income is '>50K', else 0.

    Args:
        path:
            Path to the Adult CSV file.

    Returns:
        Cleaned pandas DataFrame with an additional 'income_binary' column.
    """
    df = pd.read_csv(path, header=None)

    if df.shape[1] == len(ADULT_COLUMNS):
        df.columns = ADULT_COLUMNS
    else:
        df = pd.read_csv(path)
        if df.shape[1] != len(ADULT_COLUMNS):
            raise ValueError(f"Unexpected Adult dataset shape: {df.shape}")
        df.columns = ADULT_COLUMNS

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df = df.replace("?", np.nan).dropna().reset_index(drop=True)

    df[TARGET_COL] = df[TARGET_COL].replace({
        "<=50K.": "<=50K",
        ">50K.": ">50K",
    })

    df["income_binary"] = (df[TARGET_COL] == ">50K").astype(int)

    return df

# =========================================================
# Preprocessing
# =========================================================
def preprocess_adult(df: pd.DataFrame, test_size: float, random_state: int):
    """
    Split and preprocess the Adult dataset for neural-network training.

    Preprocessing steps:
      - Split the dataset into train and test sets using stratified sampling.
      - Standardize numerical columns using StandardScaler.
      - One-hot encode categorical columns using OneHotEncoder.
      - Convert all features to float32 for TensorFlow.
      - Convert labels to float32 column vectors for binary classification.

    Why stratified split?
      The Adult dataset is imbalanced, so stratification keeps the same income
      class ratio in both training and testing sets.

    Args:
        df:
            Cleaned Adult DataFrame.
        test_size:
            Fraction of records used for testing.
        random_state:
            Random seed for reproducibility.

    Returns:
        X_train_np, X_test_np, y_train_np, y_test_np:
            NumPy arrays ready for TensorFlow training.
    """
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["income_binary"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )

    X_train_np = preprocessor.fit_transform(X_train).astype("float32")
    X_test_np = preprocessor.transform(X_test).astype("float32")

    y_train_np = y_train.to_numpy().astype("float32").reshape(-1, 1)
    y_test_np = y_test.to_numpy().astype("float32").reshape(-1, 1)
    
    return X_train_np, X_test_np, y_train_np, y_test_np

# =========================================================
# Model definition
# =========================================================
def create_mlp(input_dim: int) -> tf.keras.Model:
    """
    Create a simple Multi-Layer Perceptron for Adult income classification.

    Architecture:
      - Input layer with dimension equal to the preprocessed feature vector.
      - Dense hidden layer with 32 ReLU units.
      - Dense hidden layer with 16 ReLU units.
      - Sigmoid output layer for binary classification.

    The sigmoid output represents the probability that income is '>50K'.

    Args:
        input_dim:
            Number of preprocessed input features.

    Returns:
        A compiled-ready TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
    return model

# =========================================================
# Privacy accounting
# =========================================================
def get_epsilon(num_examples, batch_size, noise_multiplier, epochs, delta):
    """
    Compute the differential privacy epsilon value for DP-SGD.

    This function uses TensorFlow Privacy's privacy accountant.

    It extracts two epsilon estimates from the privacy statement:
      1. epsilon_conservative:
         Assumes each example occurs once per epoch.
      2. epsilon_poisson:
         Assumes Poisson sampling.

    In the saved CSV, epsilon_conservative is used as the main epsilon value.

    Args:
        num_examples:
            Number of training records.
        batch_size:
            DP-SGD batch size.
        noise_multiplier:
            Amount of Gaussian noise added to clipped gradients.
        epochs:
            Number of training epochs.
        delta:
            DP delta parameter.

    Returns:
        epsilon_conservative:
            Conservative epsilon estimate.
        epsilon_poisson:
            Poisson-sampling epsilon estimate.
        statement:
            Full TensorFlow Privacy text statement.
    """
    try:
        statement = compute_dp_sgd_privacy_statement(
            number_of_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            noise_multiplier=noise_multiplier,
            delta=delta,
            used_microbatching=True,
        )
    except TypeError:
        # Compatibility fallback for older/newer TensorFlow Privacy signatures.
        statement = compute_dp_sgd_privacy_statement(
            num_examples,
            batch_size,
            epochs,
            noise_multiplier,
            delta,
            used_microbatching=True,
        )

    statement = str(statement)

    conservative_match = re.search(
        r"Epsilon with each example occurring once per epoch:\s*([0-9.]+)",
        statement,
    )

    poisson_match = re.search(
        r"Epsilon assuming Poisson sampling \(\*\):\s*([0-9.]+)",
        statement,
    )

    epsilon_conservative = (
        float(conservative_match.group(1)) if conservative_match else np.nan
    )

    epsilon_poisson = (
        float(poisson_match.group(1)) if poisson_match else np.nan
    )

    return epsilon_conservative, epsilon_poisson, statement

# =========================================================
# Model evaluation
# =========================================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained binary classifier on the Adult test set.

    Metrics:
      - Accuracy
      - Misclassification rate
      - Precision
      - Recall
      - AUC

    The model outputs probabilities from the sigmoid layer.
    These probabilities are thresholded at 0.5 to produce binary predictions.

    The debug prints help detect unstable DP training behavior, such as:
      - NaN predictions
      - infinite predictions
      - probabilities outside the expected range

    Args:
        model:
            Trained TensorFlow Keras model.
        X_test:
            Preprocessed test features.
        y_test:
            Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    probs = model.predict(X_test, verbose=0).reshape(-1)

    print("Prediction debug:")
    print("  has_nan:", np.isnan(probs).any())
    print("  has_inf:", np.isinf(probs).any())
    print("  min:", np.nanmin(probs))
    print("  max:", np.nanmax(probs))

    # Safety cleanup in case DP training produces unstable predictions.
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)

    y_true = np.asarray(y_test).reshape(-1).astype(int)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "misclassification_rate": round(float(1.0 - accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, probs)), 4),
    }

# =========================================================
# Baseline training
# =========================================================
def train_baseline(X_train, y_train, X_test, y_test, args):
    """
    Train a non-private baseline MLP.

    This model uses ordinary SGD without gradient clipping or noise.
    It is used as the reference point for measuring the utility loss caused
    by DP-SGD.

    Args:
        X_train, y_train:
            Training data.
        X_test, y_test:
            Test data.
        args:
            Parsed command-line arguments.

    Returns:
        Dictionary containing training configuration and evaluation metrics.
    """
    model = create_mlp(X_train.shape[1])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    start = time.time()
    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=2,
    )
    train_time = time.time() - start

    metrics = evaluate_model(model, X_test, y_test)

    return {
        "setting": "adult_mlp_baseline",
        "noise_multiplier": 0.0,
        "epsilon": np.nan,
        "delta": args.delta,
        "l2_norm_clip": np.nan,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_time_sec": round(train_time, 2),
        **metrics,
    }

# =========================================================
# DP-SGD training
# =========================================================
def train_dp_sgd(X_train, y_train, X_test, y_test, args, noise_multiplier):
    """
    Train an MLP using Differentially Private SGD.

    DP-SGD modifies ordinary SGD in two key ways:
      1. Per-example gradient clipping:
         Each training example's gradient is clipped to a maximum L2 norm.
         This limits the influence of any single record.
      2. Gaussian noise addition:
         Noise is added to the clipped gradients before updating model weights.
         This creates the differential privacy guarantee.

    Important parameters:
      - l2_norm_clip controls the clipping threshold.
      - noise_multiplier controls how much Gaussian noise is added.
      - epsilon is computed after training using TensorFlow Privacy.

    Args:
        X_train, y_train:
            Training data.
        X_test, y_test:
            Test data.
        args:
            Parsed command-line arguments.
        noise_multiplier:
            Gaussian noise multiplier for DP-SGD.

    Returns:
        Dictionary containing training configuration, privacy values, and
        optionally evaluation metrics.
    """
    model = create_mlp(X_train.shape[1])

    # DP optimizers require unreduced per-example loss.
    # This allows the optimizer to clip gradients per example or per microbatch.
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE,
    )
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=args.l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=args.num_microbatches,
        learning_rate=args.learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    # drop_remainder=True is important for DP-SGD because the optimizer expects
    # fixed batch sizes that divide cleanly into microbatches.
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=args.random_state)
    train_ds = train_ds.batch(args.batch_size, drop_remainder=True)
    
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(args.batch_size)
    
    start = time.time()
    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        verbose=2,
    )
    train_time = time.time() - start

    epsilon_conservative, epsilon_poisson, statement = get_epsilon(
        num_examples=len(X_train),
        batch_size=args.batch_size,
        noise_multiplier=noise_multiplier,
        epochs=args.epochs,
        delta=args.delta,
    )
    metrics = evaluate_model(model, X_test, y_test)

    return {
        "setting": "adult_mlp_dp_sgd",
        "noise_multiplier": noise_multiplier,
        "epsilon_conservative": epsilon_conservative,
        "epsilon_poisson": epsilon_poisson,
        "epsilon": epsilon_conservative,
        "delta": args.delta,
        "l2_norm_clip": args.l2_norm_clip,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_time_sec": round(train_time, 2),
        #"privacy_statement": statement,
        #**metrics,
    }

# =========================================================
# Main experiment driver
# =========================================================
def main():
    """
    Main entry point.

    Steps:
      1. Parse command-line arguments.
      2. Validate DP-SGD microbatch configuration.
      3. Set random seeds for reproducibility.
      4. Load and preprocess the Adult dataset.
      5. Train a non-private baseline MLP.
      6. Train DP-SGD MLP models for several noise multipliers.
      7. Save all results to CSV.
      8. Save experiment configuration to JSON.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="hw3_adult_dpsgd_outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--num_microbatches", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--l2_norm_clip", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--noise_multipliers", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # DP-SGD requires batch size to be divisible by the number of microbatches.
    # With num_microbatches=batch_size, each microbatch contains one example.
    if args.batch_size % args.num_microbatches != 0:
        raise ValueError("batch_size must be divisible by num_microbatches")

    tf.random.set_seed(args.random_state)
    np.random.seed(args.random_state)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataframe(args.adult_path)
    X_train, X_test, y_train, y_test = preprocess_adult(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    results = []

    print("\n" + "=" * 80)
    print("Training Adult MLP baseline")
    print("=" * 80)
    baseline_result = train_baseline(X_train, y_train, X_test, y_test, args)
    results.append(baseline_result)

    for noise in args.noise_multipliers:
        print("\n" + "=" * 80)
        print(f"Training Adult MLP with DP-SGD | noise_multiplier={noise}")
        print("=" * 80)

        result = train_dp_sgd(
            X_train,
            y_train,
            X_test,
            y_test,
            args,
            noise_multiplier=noise,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(outdir / "adult_dpsgd_mlp_metrics.csv", index=False)

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nSaved outputs to:", outdir.resolve())
    print(results_df)


if __name__ == "__main__":
    main()
