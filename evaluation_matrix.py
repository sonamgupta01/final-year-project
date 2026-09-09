#!/usr/bin/env python3
"""
NoC Hotspot Detection - Generic Evaluation Framework
====================================================

A dataset-independent evaluation module for Network-on-Chip hotspot detection systems.
This module provides standardized metrics and comparison capabilities for academic evaluation.

Author: NoC Hotspot Detection Project
Version: 1.2
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS_FOLDER = "datasets"
OUTPUT_FOLDER = "evaluation_results"
METRICS_COLUMNS = [
    "Dataset", "Total_Samples", "Hotspot_Count", "Hotspot_Percentage",
    "Avg_Latency", "Avg_Throughput", "Avg_Network_Load",
    "Congestion_Score_Mean", "Congestion_Score_Std",
    "Latency_Threshold", "Throughput_Threshold",
    "Packet_Delivery_Ratio", "Efficiency_Score",
    "Response_Time_Est", "Accuracy_Proxy"
]


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_latency_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate latency-related metrics."""
    if 'avg_latency' not in df.columns:
        return {"avg_latency": 0, "latency_std": 0, "latency_threshold": 0}
    
    return {
        "avg_latency": df['avg_latency'].mean(),
        "latency_std": df['avg_latency'].std(),
        "latency_threshold": df['avg_latency'].quantile(0.80)
    }


def calculate_throughput_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate throughput-related metrics."""
    if 'throughput' not in df.columns:
        return {"avg_throughput": 0, "throughput_std": 0, "throughput_threshold": 0}
    
    return {
        "avg_throughput": df['throughput'].mean(),
        "throughput_std": df['throughput'].std(),
        "throughput_threshold": df['throughput'].quantile(0.20)
    }


def calculate_congestion_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate congestion-related metrics."""
    if 'congestion_score' not in df.columns:
        return {"congestion_mean": 0, "congestion_std": 0}
    
    return {
        "congestion_mean": df['congestion_score'].mean(),
        "congestion_std": df['congestion_score'].std()
    }


def calculate_hotspot_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate hotspot detection metrics."""
    if 'hotspot_detected' not in df.columns:
        return {"hotspot_count": 0, "hotspot_percentage": 0}
    
    hotspot_count = df['hotspot_detected'].sum()
    total = len(df)
    
    return {
        "hotspot_count": int(hotspot_count),
        "hotspot_percentage": (hotspot_count / total * 100) if total > 0 else 0
    }


def calculate_network_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate network performance metrics."""
    metrics = {}
    
    # Network load
    if 'network_load' in df.columns:
        metrics["avg_network_load"] = df['network_load'].mean()
    else:
        metrics["avg_network_load"] = 0
    
    # Packet delivery ratio (proxy: throughput / network_load)
    if 'throughput' in df.columns and 'network_load' in df.columns:
        if df['network_load'].mean() > 0:
            metrics["packet_delivery_ratio"] = df['throughput'].mean() / df['network_load'].mean()
        else:
            metrics["packet_delivery_ratio"] = 0
    else:
        metrics["packet_delivery_ratio"] = 0
    
    # Efficiency score (inverse of congestion)
    if 'congestion_score' in df.columns:
        metrics["efficiency_score"] = 1 - df['congestion_score'].mean()
    else:
        metrics["efficiency_score"] = 0
    
    # Response time estimate (based on latency)
    if 'avg_latency' in df.columns:
        metrics["response_time_est"] = df['avg_latency'].mean()
    else:
        metrics["response_time_est"] = 0
    
    # Accuracy proxy (for datasets with ground truth)
    if 'hotspot_detected' in df.columns:
        metrics["accuracy_proxy"] = 1 - abs(df['hotspot_detected'].mean() - 0.18)
    else:
        metrics["accuracy_proxy"] = 0
    
    return metrics


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(filepath: str) -> Optional[pd.DataFrame]:
    """Load a dataset from file."""
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.txt'):
            for sep in [r'\s+', ',', '\t']:
                try:
                    df = pd.read_csv(filepath, sep=sep, engine='python', header=None)
                    if len(df.columns) >= 3:
                        if df.shape[1] == 3:
                            df.columns = ['clock_cycle', 'src_node', 'dst_node']
                            return df
                        return df
                except:
                    continue
        return None
    except Exception as e:
        print(f"  Warning: Could not load {filepath}: {e}")
        return None


def process_external_trace(df: pd.DataFrame) -> pd.DataFrame:
    """Process external trace data to compute metrics similar to BookSim format."""
    if 'clock_cycle' not in df.columns:
        return df
    
    time_window = 1000
    max_time = df['clock_cycle'].max()
    time_bins = list(range(0, int(max_time) + time_window, time_window))
    
    processed_data = []
    for i, start_time in enumerate(time_bins[:-1]):
        end_time = time_bins[i + 1]
        window_data = df[
            (df['clock_cycle'] >= start_time) &
            (df['clock_cycle'] < end_time)
        ]
        
        total_packets = len(window_data)
        node_packets = {node: 0 for node in range(64)}
        
        for _, packet in window_data.iterrows():
            src = int(packet['src_node'])
            dst = int(packet['dst_node'])
            if src in node_packets:
                node_packets[src] += 1
            if dst in node_packets:
                node_packets[dst] += 1
        
        if total_packets > 0:
            avg_density = np.mean(list(node_packets.values())) / total_packets
            max_density = max(node_packets.values()) / total_packets
        else:
            avg_density = 0
            max_density = 0
        
        congestion_score = min(1.0, max_density * 0.5)
        hotspot_detected = 1 if congestion_score > 0.15 else 0
        
        processed_data.append({
            'step': i + 1,
            'total_packets': total_packets,
            'avg_node_density': avg_density,
            'max_node_density': max_density,
            'congestion_score': congestion_score,
            'hotspot_detected': hotspot_detected,
            'traffic_pattern': 'external_trace'
        })
    
    return pd.DataFrame(processed_data)


def discover_datasets(folder: str = DATASETS_FOLDER) -> List[str]:
    """Discover all dataset files in the folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []
    
    valid_extensions = ['.csv', '.txt']
    datasets = []
    
    for filename in os.listdir(folder):
        if any(filename.endswith(ext) for ext in valid_extensions):
            datasets.append(os.path.join(folder, filename))
    
    return sorted(datasets)


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

def evaluate_dataset(filepath: str) -> Dict[str, any]:
    """Evaluate a single dataset and return metrics."""
    df = load_dataset(filepath)
    
    if df is None or len(df) == 0:
        return {"error": f"Could not load or empty dataset: {filepath}"}
    
    dataset_name = os.path.basename(filepath)
    
    # Check if it's an external trace and process it
    if 'clock_cycle' in df.columns:
        df = process_external_trace(df)
    
    results = {"Dataset": dataset_name, "Total_Samples": len(df)}
    
    # Hotspot metrics
    hotspot_metrics = calculate_hotspot_metrics(df)
    results.update(hotspot_metrics)
    
    # Latency metrics
    latency_metrics = calculate_latency_metrics(df)
    results["Avg_Latency"] = latency_metrics["avg_latency"]
    results["Latency_Threshold"] = latency_metrics["latency_threshold"]
    
    # Throughput metrics
    throughput_metrics = calculate_throughput_metrics(df)
    results["Avg_Throughput"] = throughput_metrics["avg_throughput"]
    results["Throughput_Threshold"] = throughput_metrics["throughput_threshold"]
    
    # Congestion metrics
    congestion_metrics = calculate_congestion_metrics(df)
    results["Congestion_Score_Mean"] = congestion_metrics["congestion_mean"]
    results["Congestion_Score_Std"] = congestion_metrics["congestion_std"]
    
    # Network metrics
    network_metrics = calculate_network_metrics(df)
    results["Packet_Delivery_Ratio"] = network_metrics["packet_delivery_ratio"]
    results["Efficiency_Score"] = network_metrics["efficiency_score"]
    results["Response_Time_Est"] = network_metrics["response_time_est"]
    results["Accuracy_Proxy"] = network_metrics["accuracy_proxy"]
    
    return results


def run_evaluation(datasets_folder: str = DATASETS_FOLDER) -> pd.DataFrame:
    """Run evaluation on all datasets in the folder."""
    print("=" * 70)
    print(" NoC Hotspot Detection - Generic Evaluation Framework")
    print("=" * 70)
    
    datasets = discover_datasets(datasets_folder)
    
    if not datasets:
        print(f"\nNo datasets found in '{datasets_folder}' folder.")
        print("Please add dataset files (CSV or TXT) to the folder.")
        return pd.DataFrame()
    
    print(f"\nFound {len(datasets)} dataset(s) to evaluate:\n")
    for ds in datasets:
        print(f"  - {os.path.basename(ds)}")
    
    print("\n" + "-" * 70)
    print(" Running Evaluation...")
    print("-" * 70 + "\n")
    
    results = []
    for filepath in datasets:
        print(f"Evaluating: {os.path.basename(filepath)}...")
        result = evaluate_dataset(filepath)
        if "error" not in result:
            results.append(result)
            print(f"  ✓ Completed - {result['Total_Samples']} samples")
        else:
            print(f"  ✗ {result['error']}")
    
    if not results:
        print("\nNo valid results generated.")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    column_order = [col for col in METRICS_COLUMNS if col in df_results.columns]
    df_results = df_results[column_order]
    
    return df_results


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def print_results_table(df: pd.DataFrame):
    """Print results in a formatted table."""
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS MATRIX")
    print("=" * 70)
    
    if df.empty:
        print("No results to display.")
        return
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + df.to_string(index=False))
    
    print("\n" + "-" * 70)
    print(" SUMMARY STATISTICS")
    print("-" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "Dataset":
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  {col}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")


def save_results(df: pd.DataFrame, output_folder: str = OUTPUT_FOLDER):
    """Save results to CSV and generate visualizations."""
    if df.empty:
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_folder, f"evaluation_matrix_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    try:
        generate_comparison_chart(df, output_folder, timestamp)
    except Exception as e:
        print(f"Warning: Could not generate chart: {e}")


def generate_comparison_chart(df: pd.DataFrame, output_folder: str, timestamp: str):
    """Generate comparison visualization charts."""
    if df.empty or len(df) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Hotspot Percentage Comparison
    ax1 = axes[0, 0]
    if 'Hotspot_Percentage' in df.columns:
        bars1 = ax1.bar(range(len(df)), df['Hotspot_Percentage'], color='coral', alpha=0.7)
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['Dataset'].str[:10], rotation=45, ha='right')
        ax1.set_ylabel('Hotspot Percentage (%)')
        ax1.set_title('Hotspot Detection Rate by Dataset')
        ax1.grid(True, alpha=0.3)
    
    # 2. Latency vs Throughput
    ax2 = axes[0, 1]
    if 'Avg_Latency' in df.columns and 'Avg_Throughput' in df.columns:
        ax2.scatter(df['Avg_Latency'], df['Avg_Throughput'], s=100, alpha=0.7, c='blue')
        for i, txt in enumerate(df['Dataset'].str[:8]):
            ax2.annotate(txt, (df['Avg_Latency'].iloc</tool_call>