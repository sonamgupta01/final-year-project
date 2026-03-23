#!/usr/bin/env python3
"""
Phase-2: Continuous Hotspot Prediction + Adaptive Rerouting
Aligned strictly with Phase-1 results.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "lstm_hotspot_model.h5"
DATASET_FILE = "booksim_dataset_raw.csv"

WINDOW_SIZE = 10
TOP_K_NODES = 3
HOTSPOT_THRESHOLD = 0.5

TRAFFIC_REROUTE_PERCENT = {
    "Mild": 0.20,
    "Moderate": 0.40,
    "Severe": 0.60
}

FEATURE_COLS = [
    "injection_rate",
    "network_load",
    "throughput",
    "avg_latency",
    "network_latency",
    "unstable"
]

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("="*60)
    print(" LOADING TRAINED LSTM MODEL")
    print("="*60)
    model = keras.models.load_model(MODEL_FILE)
    print("✓ Model loaded")
    print("✓ Input shape:", model.input_shape)
    print("✓ Output shape:", model.output_shape)
    return model

# ============================================================
# NORMALIZATION
# ============================================================

def get_stats(df):
    stats = {}
    for col in FEATURE_COLS:
        stats[col] = {
            "min": df[col].min(),
            "max": df[col].max()
        }
    return stats

def normalize_row(row, stats):
    vals = []
    for col in FEATURE_COLS:
        mn = stats[col]["min"]
        mx = stats[col]["max"]
        vals.append((row[col]-mn)/(mx-mn) if mx>mn else 0)
    return vals

def prepare_window(df, start, stats):
    window = df.iloc[start:start+WINDOW_SIZE]
    X = []
    for _, r in window.iterrows():
        X.append(normalize_row(r, stats))
    X = np.array(X).reshape(1, WINDOW_SIZE, len(FEATURE_COLS))
    return X, window

# ============================================================
# WHEN PREDICTION
# ============================================================

def predict_when(model, X):
    prob = model.predict(X, verbose=0)[0][0]
    label = "HOTSPOT" if prob >= HOTSPOT_THRESHOLD else "NO HOTSPOT"
    return prob, label

# ============================================================
# PHASE-1 BASED SEVERITY
# ============================================================

def get_severity_from_phase1(window_df):
    """
    Severity derived from Phase-1 congestion_score.
    """
    avg_congestion = window_df["congestion_score"].mean()

    if avg_congestion >= 0.55:
        return "Severe"
    elif avg_congestion >= 0.45:
        return "Moderate"
    else:
        return "Mild"

# ============================================================
# WHERE PREDICTION (STRICT PHASE-1 MATCH)
# ============================================================

def predict_hotspot_nodes_from_data(window_df):
    nodes = []

    # First: Use Phase-1 hotspot_nodes column
    if "hotspot_nodes" in window_df.columns:
        for val in window_df["hotspot_nodes"]:
            if pd.isna(val):
                continue
            clean = str(val).replace("[","").replace("]","")
            parts = clean.split(",")
            for p in parts:
                try:
                    nodes.append(int(p.strip()))
                except:
                    pass

    # If empty → fallback using highest congestion_score rows
    if len(nodes) == 0:
        top_rows = window_df.sort_values(
            by="congestion_score",
            ascending=False
        ).head(TOP_K_NODES)

        for s in top_rows["step"]:
            nodes.append(int(s % 64))

    return list(set(nodes))[:TOP_K_NODES]

# ============================================================
# NEIGHBOR LOGIC
# ============================================================

def get_1hop_neighbors(node, mesh=8):
    r = node // mesh
    c = node % mesh
    neighbors = []

    if r > 0: neighbors.append(node - mesh)
    if r < mesh-1: neighbors.append(node + mesh)
    if c > 0: neighbors.append(node - 1)
    if c < mesh-1: neighbors.append(node + 1)

    return neighbors

# ============================================================
# REROUTING LOGIC
# ============================================================

def simulate_rerouting(nodes, severity, window_df):

    reroute_percent = TRAFFIC_REROUTE_PERCENT[severity]

    # Estimate load using congestion_score
    window_df["node_id"] = window_df["step"] % 64
    node_load = window_df.groupby("node_id")["congestion_score"].mean().to_dict()

    plans = []

    for node in nodes:
        neighbors = get_1hop_neighbors(node)

        # Sort neighbors by least congestion
        neighbors = sorted(neighbors, key=lambda x: node_load.get(x, 0))

        if len(neighbors) >= 2:
            plans.append((node, neighbors[0], neighbors[1], reroute_percent/2))

    reduced_congestion = window_df["congestion_score"].mean() * (1 - reroute_percent)

    return plans, reroute_percent, reduced_congestion

# ============================================================
# PRINT FUNCTIONS
# ============================================================

def print_window(win, start, end, prob, label, severity, nodes):
    print("-"*60)
    print(f"Window {win} | Steps {start}-{end}")
    print(f"Model Probability : {prob:.4f}")
    print(f"Prediction        : {label}")

    if label == "HOTSPOT":
        print(f"Phase-1 Severity  : {severity}")
        print(f"Hotspot Nodes     : {nodes}")
    else:
        print("Status            : Normal Traffic")

    print("-"*60)

def print_rerouting(severity, plans, reroute_percent, reduced):

    if severity == "Mild":
        print("Mild hotspot → Managed via buffering. No rerouting required.")
        print("="*60)
        return

    print("➡️ Adaptive Rerouting Activated")
    print(f"Rerouting Percent : {reroute_percent*100:.0f}%")

    for h, n1, n2, p in plans:
        print(f"- Node {h} → {n1} , {n2} ({p*100:.0f}% each)")

    print(f"Post-Rerouting Congestion : {reduced:.2f}")
    print("="*60)

# ============================================================
# MAIN LOOP
# ============================================================

def run(model, df):

    stats = get_stats(df)
    start = 0
    win = 1
    results = []

    while start + WINDOW_SIZE <= len(df):

        X, win_df = prepare_window(df, start, stats)
        prob, label = predict_when(model, X)

        if label == "HOTSPOT":
            severity = get_severity_from_phase1(win_df)
            nodes = predict_hotspot_nodes_from_data(win_df)
        else:
            severity = None
            nodes = []

        print_window(win, start+1, start+WINDOW_SIZE,
                     prob, label, severity, nodes)

        if label == "HOTSPOT":
            if len(nodes) == 0:
                print("⚠ Early hotspot warning detected.")
                print("No dominant node identified yet.")
                print("Action: Monitoring mode activated (no rerouting).")
                print("="*60)
            else:
                plans, reroute_percent, reduced = simulate_rerouting(nodes, severity, win_df)
                print_rerouting(severity, plans, reroute_percent, reduced)

        results.append(label)

        start += WINDOW_SIZE
        win += 1

    print("\nTotal windows :", len(results))
    print("Hotspot windows :", results.count("HOTSPOT"))

# ============================================================
# ENTRY
# ============================================================

def main():
    print("="*60)
    print(" 🔮 PHASE-2 HOTSPOT PREDICTION SYSTEM")
    print("="*60)

    model = load_model()

    print("\nLOADING DATASET")
    df = pd.read_csv(DATASET_FILE)
    print("✓ Loaded", len(df), "timesteps")

    run(model, df)

if __name__ == "__main__":
    main()