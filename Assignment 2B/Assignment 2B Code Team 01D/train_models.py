"""
COS30019 Assignment 2B: Traffic-Based Route Guidance System
ML Training Module

Trains and evaluates three deep learning models for traffic flow prediction:
  - LSTM  : Long Short-Term Memory (mandatory)
  - GRU   : Gated Recurrent Unit (mandatory)
  - CNN-LSTM : Convolutional + LSTM hybrid (custom choice)

All models are trained on the same data split for a fair comparison.
Evaluation metrics: MAE, RMSE, MAPE.

Team ID: 01D
Team Members: Saniru, Haresh, Manula

Usage:
    python train_models.py
    (Requires data_processor.py to have been run first)
"""

import os
import time
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warning logs
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout,
    Conv1D, MaxPooling1D, Flatten,
    Input, Reshape, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ────────────────────────────
# PATHS & HYPER-PARAMETERS
# ────────────────────────────

BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEQUENCE_LEN = 12      # must match data_processor.py
EPOCHS       = 50
BATCH_SIZE   = 64
LEARNING_RATE = 0.001
PATIENCE     = 10      # early-stopping patience


# ────────────────────────
# LOAD PROCESSED DATA
# ────────────────────────

def load_data():
    """Load the .npy arrays and scaler saved by data_processor.py."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Reshape for Keras: (samples, timesteps, features=1)
    X_train = X_train.reshape(-1, SEQUENCE_LEN, 1)
    X_test  = X_test.reshape(-1,  SEQUENCE_LEN, 1)
    y_train = y_train.reshape(-1, 1)
    y_test  = y_test.reshape(-1,  1)

    print(f"  Loaded  X_train={X_train.shape}  X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler


# ────────────────────────
# MODEL DEFINITIONS
# ────────────────────────

def build_lstm(seq_len: int = SEQUENCE_LEN) -> Sequential:
    """
    LSTM model for sequential traffic flow prediction.

    Architecture:
      LSTM(64, return_sequences=True) → Dropout(0.2)
      LSTM(32)                        → Dropout(0.2)
      Dense(16, relu)                 → Dense(1)

    The stacked LSTM captures both short-range fluctuations and
    longer hourly/daily traffic patterns.
    Loss: MSE  |  Optimiser: Adam
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="LSTM")

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="mse",
                  metrics=["mae"])
    return model


def build_gru(seq_len: int = SEQUENCE_LEN) -> Sequential:
    """
    GRU model — same topology as LSTM for a controlled comparison.

    GRU uses fewer parameters (no output gate) so it is typically
    faster to train while achieving comparable accuracy on short series.
    Loss: MSE  |  Optimiser: Adam
    """
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="GRU")

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="mse",
                  metrics=["mae"])
    return model


def build_cnn_lstm(seq_len: int = SEQUENCE_LEN) -> Sequential:
    """
    CNN-LSTM hybrid model (custom third technique).

    The 1-D convolutional front-end extracts local spatial features
    from the input window (e.g., sharp morning peaks), which the LSTM
    then models as a compressed temporal sequence.

    Architecture:
      Conv1D(32, kernel=3, relu) → Conv1D(64, kernel=3, relu)
      MaxPooling1D(2)             → LSTM(32)
      Dropout(0.2)                → Dense(16, relu) → Dense(1)

    This architecture has been shown to outperform standalone LSTM/GRU
    on traffic data with strong periodic patterns
    (Zhao et al., 2017; Wu & Tan, 2016).
    Loss: MSE  |  Optimiser: Adam
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu",
               padding="same", input_shape=(seq_len, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="CNN_LSTM")

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="mse",
                  metrics=["mae"])
    return model


# ──────────────────────────
# EVALUATION METRICS
# ──────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    scaler) -> dict:
    """
    Inverse-transform predictions back to vehicle counts and compute:
      MAE  — Mean Absolute Error
      RMSE — Root Mean Squared Error
      MAPE — Mean Absolute Percentage Error  (avoids div-by-zero)
    """
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mae  = float(np.mean(np.abs(y_true_inv - y_pred_inv)))
    rmse = float(np.sqrt(np.mean((y_true_inv - y_pred_inv) ** 2)))

    # MAPE: skip zero actuals to avoid division by zero
    mask = y_true_inv != 0
    mape = float(np.mean(np.abs((y_true_inv[mask] - y_pred_inv[mask])
                                 / y_true_inv[mask])) * 100)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4)}


# ──────────────────────
# TRAIN ONE MODEL
# ──────────────────────

def train_model(model: Sequential,
                X_train, y_train, X_test, y_test,
                scaler,
                epochs: int    = EPOCHS,
                batch_size: int = BATCH_SIZE) -> dict:
    """
    Train a Keras model with early stopping and best-model checkpointing.

    Returns a result dict with: history, predictions, metrics, train_time.
    """
    name = model.name
    ckpt_path = os.path.join(MODEL_DIR, f"{name}_best.keras")
    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]

    print(f"\n  Training {name} …")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.time() - t0

    # Evaluate on held-out test set
    y_pred = model.predict(X_test, verbose=0)
    metrics = compute_metrics(y_test, y_pred, scaler)
    metrics["train_time_s"] = round(elapsed, 1)
    metrics["epochs_run"]   = len(history.history["loss"])

    print(f"    Epochs: {metrics['epochs_run']}  |  "
          f"MAE: {metrics['MAE']:.4f}  |  "
          f"RMSE: {metrics['RMSE']:.4f}  |  "
          f"MAPE: {metrics['MAPE']:.2f}%  |  "
          f"Time: {elapsed:.1f}s")

    return {
        "model":    model,
        "history":  history.history,
        "y_pred":   y_pred,
        "metrics":  metrics,
        "ckpt":     ckpt_path,
    }


# ─────────────────────
# SAVE + PLOT
# ─────────────────────

def save_results(results: dict, scaler, y_test: np.ndarray):
    """Save metrics JSON, training curves, and prediction comparison plot."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── metrics summary ──────────────────────
    summary = {name: res["metrics"] for name, res in results.items()}
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── training-loss curves ──────────────────
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        h = res["history"]
        ax.plot(h["loss"],     label="Train loss")
        ax.plot(h["val_loss"], label="Val loss", linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
    plt.suptitle("Training & Validation Loss per Model", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=120)
    plt.close()

    # ── prediction comparison (first 200 test points) ────────────────
    y_true_inv = scaler.inverse_transform(
        y_test[:200].reshape(-1, 1)).flatten()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true_inv, label="Actual", color="black", linewidth=1.5)
    colours = ["tab:blue", "tab:orange", "tab:green"]
    for (name, res), col in zip(results.items(), colours):
        y_pred_inv = scaler.inverse_transform(
            res["y_pred"][:200]).flatten()
        ax.plot(y_pred_inv, label=name, alpha=0.8, color=col, linewidth=1)
    ax.set_title("Predicted vs Actual Traffic Flow (first 200 test steps)",
                 fontweight="bold")
    ax.set_xlabel("Time step (× 15 min)")
    ax.set_ylabel("Vehicles / 15 min")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predictions_comparison.png"), dpi=120)
    plt.close()

    # ── metrics comparison bar chart ─────────
    metric_names = ["MAE", "RMSE", "MAPE"]
    model_names  = list(results.keys())
    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["tab:blue", "tab:orange", "tab:green"]
    for i, (mname, col) in enumerate(zip(model_names, colours)):
        vals = [results[mname]["metrics"][m] for m in metric_names]
        ax.bar(x + i * width, vals, width, label=mname, color=col)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.set_title("Model Comparison — MAE / RMSE / MAPE", fontweight="bold")
    ax.set_ylabel("Error value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=120)
    plt.close()

    print(f"\n  Plots saved to: {RESULTS_DIR}")


# ─────────
# MAIN
# ─────────

def run_training(progress_callback=None):
    """
    Full training pipeline.
    progress_callback(model_name, metrics) is called after each model
    so the GUI can display live updates.

    Returns results dict and scaler.
    """
    print("=" * 55)
    print("  TBRGS ML Training Pipeline")
    print("=" * 55)

    print("\n[1] Loading data …")
    X_train, y_train, X_test, y_test, scaler = load_data()

    models_to_train = {
        "LSTM":     build_lstm(),
        "GRU":      build_gru(),
        "CNN_LSTM": build_cnn_lstm(),
    }

    print("\n[2] Training models …")
    results = {}
    for name, model in models_to_train.items():
        res = train_model(model, X_train, y_train, X_test, y_test, scaler)
        results[name] = res
        if progress_callback:
            progress_callback(name, res["metrics"])

    print("\n[3] Saving results and plots …")
    save_results(results, scaler, y_test)

    # Print comparison table
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Time(s)':>9}")
    print("  " + "-" * 50)
    for name, res in results.items():
        m = res["metrics"]
        print(f"  {name:<12} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
              f"{m['MAPE']:>8.2f} {m['train_time_s']:>9.1f}")
    print("=" * 55)

    # Save the scaler alongside (needed by travel_time.py and gui.py)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("\n✓ Training complete.\n")
    return results, scaler


if __name__ == "__main__":
    run_training()
