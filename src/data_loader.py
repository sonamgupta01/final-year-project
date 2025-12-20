#!/usr/bin/env python3
"""
Modular Data Loader for NoC Hotspot Detection Framework
Supports BookSim simulator data and external trace datasets
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class DataLoader:
    """Modular data loader for different input sources"""

    def __init__(self):
        self.supported_formats = ['booksim', 'external_trace']

    def load_data(self, file_path: str, data_type: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load and process data based on type

        Args:
            file_path: Path to data file
            data_type: 'booksim' or 'external_trace'
            **kwargs: Additional parameters for processing

        Returns:
            Processed DataFrame with unified format
        """
        if data_type not in self.supported_formats:
            raise ValueError(f"Unsupported data type: {data_type}. Supported: {self.supported_formats}")

        if data_type == 'booksim':
            return self._load_booksim_data(file_path, **kwargs)
        elif data_type == 'external_trace':
            return self._load_external_trace_data(file_path, **kwargs)

    def _load_booksim_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load BookSim simulator data (existing format)"""
        df = pd.read_csv(file_path)

        # Ensure required columns exist
        required_cols = ['step', 'congestion_score', 'hotspot_detected']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"BookSim data missing required columns: {missing_cols}")

        # Add data_source column for tracking
        df['data_source'] = 'booksim'

        return df

    def _load_external_trace_data(self, file_path: str, time_window: int = 1000,
                                num_nodes: int = 64, **kwargs) -> pd.DataFrame:
        """
        Load external trace data and compute congestion metrics

        Expected trace format: clock_cycle, src_node, dst_node, [packet_size]
        """
        # Read the file line by line to handle inconsistent formatting
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        clock_cycle = int(parts[0])
                        src_node = int(parts[1])
                        dst_node = int(parts[2])
                        data.append([clock_cycle, src_node, dst_node])
                    except ValueError:
                        continue  # Skip lines that can't be parsed

        df = pd.DataFrame(data, columns=['clock_cycle', 'src_node', 'dst_node'])

        # Validate required columns
        required_cols = ['clock_cycle', 'src_node', 'dst_node']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"External trace data missing required columns: {missing_cols}")

        # Process trace data
        processed_df = self._process_trace_data(df, time_window, num_nodes)

        # Add data_source column
        processed_df['data_source'] = 'external_trace'

        return processed_df

    def _process_trace_data(self, trace_df: pd.DataFrame, time_window: int,
                          num_nodes: int) -> pd.DataFrame:
        """
        Process trace data to compute per-node packet density and congestion scores
        """
        # Create time windows
        max_time = trace_df['clock_cycle'].max()
        time_bins = list(range(0, int(max_time) + time_window, time_window))

        processed_data = []

        for i, start_time in enumerate(time_bins[:-1]):
            end_time = time_bins[i + 1]
            window_data = trace_df[
                (trace_df['clock_cycle'] >= start_time) &
                (trace_df['clock_cycle'] < end_time)
            ]

            step = i + 1

            # Compute per-node packet counts
            node_packets = {}
            for node in range(num_nodes):
                node_packets[node] = 0
            for _, packet in window_data.iterrows():
                src = int(packet['src_node'])
                dst = int(packet['dst_node'])
                if src in node_packets:
                    node_packets[src] += 1
                if dst in node_packets:
                    node_packets[dst] += 1

            # Compute density
            total_packets = len(window_data)
            if total_packets > 0:
                node_density = {node: count / total_packets for node, count in node_packets.items()}
            else:
                node_density = {node: 0.0 for node in node_packets}

            avg_density = np.mean(list(node_density.values())) if node_density else 0
            max_density = max(node_density.values()) if node_density else 0

            # Derive congestion score (similar to BookSim approach)
            # Normalize density metrics
            density_score = (max_density - avg_density) / (avg_density + 1e-6) if avg_density > 0 else 0

            # Simple congestion score based on density variation
            congestion_score = min(1.0, density_score * 0.1)  # Scale to 0-1 range

            # Determine hotspot (high congestion)
            hotspot_detected = 1 if congestion_score > 0.1 else 0

            # Detect hotspot nodes (top 3 by packet count if hotspot)
            if hotspot_detected and node_packets:
                sorted_nodes = sorted(node_packets.items(), key=lambda x: x[1], reverse=True)[:3]
                hotspot_nodes = ','.join(str(node) for node, count in sorted_nodes)
            else:
                hotspot_nodes = ''

            # Store window data
            processed_data.append({
                'step': step,
                'time_window_start': start_time,
                'time_window_end': end_time,
                'total_packets': total_packets,
                'avg_node_density': avg_density,
                'max_node_density': max_density,
                'congestion_score': congestion_score,
                'hotspot_detected': hotspot_detected,
                'hotspot_nodes': hotspot_nodes,
                'node_packets': node_packets,  # Store for hotspot node recalculation
                'node_density': node_density,  # Store for potential node-level analysis
                'traffic_pattern': 'external_trace'  # Placeholder
            })

        # Apply top-k based hotspot detection
        all_densities = [(i, d['avg_node_density']) for i, d in enumerate(processed_data)]
        all_densities.sort(key=lambda x: x[1], reverse=True)
        top_k = 100  # Top 100 windows as hotspots
        if len(all_densities) > top_k:
            threshold = all_densities[top_k - 1][1]
        else:
            threshold = all_densities[-1][1] if all_densities else 0
        hotspot_indices = set(i for i, dens in all_densities[:top_k])

        # Update hotspot detection and nodes based on top-k
        for i, d in enumerate(processed_data):
            d['hotspot_detected'] = 1 if i in hotspot_indices else 0
            # Recompute hotspot_nodes based on new detection
            if d['hotspot_detected'] and 'node_packets' in d:
                node_packets = d['node_packets']
                if node_packets:
                    sorted_nodes = sorted(node_packets.items(), key=lambda x: x[1], reverse=True)[:3]
                    d['hotspot_nodes'] = ','.join(str(node) for node, count in sorted_nodes)
                else:
                    d['hotspot_nodes'] = ''
            else:
                d['hotspot_nodes'] = ''

        result_df = pd.DataFrame(processed_data)

        # Add dummy columns to match BookSim format for compatibility
        result_df['injection_rate'] = result_df['total_packets'] / time_window
        result_df['network_load'] = result_df['avg_node_density']
        result_df['throughput'] = result_df['total_packets'] / time_window
        result_df['avg_latency'] = 50.0  # Placeholder, not computed from trace
        result_df['network_latency'] = 50.0
        result_df['unstable'] = 0

        return result_df

    def _compute_node_density(self, window_data: pd.DataFrame, num_nodes: int) -> Dict[int, float]:
        """Compute packet density for each node in the time window"""
        node_packets = {}

        # Initialize all nodes
        for node in range(num_nodes):
            node_packets[node] = 0

        # Count packets per node (src or dst)
        for _, packet in window_data.iterrows():
            src = int(packet['src_node'])
            dst = int(packet['dst_node'])

            if src in node_packets:
                node_packets[src] += 1
            if dst in node_packets:
                node_packets[dst] += 1

        # Convert to density (packets per node)
        total_packets = sum(node_packets.values())
        if total_packets > 0:
            node_density = {node: count / total_packets for node, count in node_packets.items()}
        else:
            node_density = {node: 0.0 for node in node_packets}

        return node_density

def detect_natural_hotspots_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unified hotspot detection for both data sources
    For external traces, congestion_score is already computed
    """
    if df['data_source'].iloc[0] == 'booksim':
        # Use existing BookSim logic
        return detect_natural_hotspots_booksim(df)
    else:
        # For external traces, hotspots are already detected in processing
        # But we can refine using unified logic if needed
        return df

def detect_natural_hotspots_booksim(df: pd.DataFrame) -> pd.DataFrame:
    """
    Original BookSim hotspot detection logic
    """
    # Calculate congestion score based on multiple metrics
    df = df.copy()

    # Normalize metrics for scoring (higher latency = worse, lower throughput = worse)
    df['latency_score'] = (df['avg_latency'] - df['avg_latency'].min()) / (df['avg_latency'].max() - df['avg_latency'].min())
    df['throughput_score'] = 1 - ((df['throughput'] - df['throughput'].min()) / (df['throughput'].max() - df['throughput'].min()))
    df['load_efficiency'] = df['throughput'] / (df['network_load'] + 1e-6)  # Avoid division by zero
    df['efficiency_score'] = 1 - ((df['load_efficiency'] - df['load_efficiency'].min()) / (df['load_efficiency'].max() - df['load_efficiency'].min()))

    # Combined congestion score (0-1, higher = more congested)
    df['congestion_score'] = (0.4 * df['latency_score'] +
                              0.3 * df['throughput_score'] +
                              0.2 * df['efficiency_score'] +
                              0.1 * df['unstable'])

    # Detect hotspots using statistical thresholds
    # Use 75th percentile as threshold for hotspot detection
    congestion_threshold = df['congestion_score'].quantile(0.75)
    latency_threshold = df['avg_latency'].quantile(0.80)  # Top 20% latency

    # A sample is considered a hotspot if:
    # 1. High congestion score (top 25%)
    # 2. High latency (top 20%)
    # 3. OR marked as unstable
    df['hotspot_detected'] = (
        (df['congestion_score'] >= congestion_threshold) &
        (df['avg_latency'] >= latency_threshold)
    ) | (df['unstable'] == 1)

    # Convert boolean to int for consistency
    df['hotspot_detected'] = df['hotspot_detected'].astype(int)

    # Node-level hotspot detection (placeholder for external)
    def detect_node_hotspots(row):
        if row['data_source'] == 'external_trace':
            return ''  # Not implemented for external traces
        # BookSim logic
        return ''  # Simplified

    df['hotspot_nodes'] = df.apply(detect_node_hotspots, axis=1)

    print(f"Congestion score threshold: {congestion_threshold:.3f}")
    print(f"Latency threshold: {latency_threshold:.1f}")
    print(f"Hotspots detected: {df['hotspot_detected'].sum()} / {len(df)} samples")

    # Clean up temporary columns
    df = df.drop(['latency_score', 'throughput_score', 'load_efficiency', 'efficiency_score'], axis=1, errors='ignore')

    return df