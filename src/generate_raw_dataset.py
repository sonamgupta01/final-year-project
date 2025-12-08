#!/usr/bin/env python3
"""
Generate comprehensive BookSim2 dataset for NATURAL hotspot detection.
Uses only natural traffic patterns - NO manual hotspot forcing.
Hotspots will emerge naturally from traffic patterns and injection rates.
"""

import subprocess
import os
import re
import time
import pandas as pd
import random
import stat
import numpy as np

def update_config(config_file, traffic_pattern, injection_rate):
    """Update configuration with specified parameters."""
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    with open(config_file, 'w') as f:
        for line in lines:
            if line.strip().startswith('injection_rate'):
                f.write(f'injection_rate = {injection_rate:.6f};\n')
            elif line.strip().startswith('traffic'):
                f.write(f'traffic = {traffic_pattern};\n')
            else:
                f.write(line)

def run_simulation(config_file, output_file, timeout=60):
    """Run a single BookSim simulation."""
    possible = []
    env_path = os.environ.get('BOOKSIM_PATH')
    if env_path:
        possible.append(env_path)

    possible.extend([
        './booksim',
        './booksim2/src/booksim',
        os.path.join(os.path.dirname(__file__), 'booksim2', 'src', 'booksim'),
        os.path.join(os.path.dirname(__file__), '..', 'booksim2', 'src', 'booksim')
    ])

    booksim_exec = None
    for p in possible:
        if p and os.path.exists(p):
            booksim_exec = p
            break

    if not booksim_exec:
        print(f"    Error: BookSim executable not found")
        return False

    try:
        if not os.access(booksim_exec, os.X_OK):
            st = os.stat(booksim_exec)
            os.chmod(booksim_exec, st.st_mode | stat.S_IXUSR)
    except Exception as e:
        print(f"    Warning: {e}")

    config_file_abs = os.path.abspath(config_file)
    cmd = [booksim_exec, config_file_abs]
    try:
        with open(output_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("    Error: timeout")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False

def extract_metrics(output_file):
    """Extract key metrics from simulation output."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        metrics = {}
        
        match = re.search(r'Packet latency average\s*=\s*([\d.]+)', content)
        if match:
            metrics['avg_latency'] = float(match.group(1))
        
        match = re.search(r'Network latency average\s*=\s*([\d.]+)', content)
        if match:
            metrics['network_latency'] = float(match.group(1))
        
        match = re.search(r'Accepted flit rate average\s*=\s*([\d.]+)', content)
        if match:
            metrics['throughput'] = float(match.group(1))
        
        match = re.search(r'Injected flit rate average\s*=\s*([\d.]+)', content)
        if match:
            metrics['network_load'] = float(match.group(1))
        
        metrics['unstable'] = 'unstable' in content.lower() or 'exceeded' in content.lower()
        
        return metrics
    except Exception as e:
        print(f"    Error extracting metrics: {e}")
        return None

def generate_scenario_samples(config_file, scenario_name, params_list, start_idx=1):
    """Generate samples for a specific scenario WITHOUT pre-defined labels."""
    data = []
    
    print(f"\n{'='*70}")
    print(f" Generating {len(params_list)} samples for: {scenario_name}")
    print(f"{'='*70}")
    
    for i, params in enumerate(params_list, start_idx):
        traffic = params['traffic']
        rate = params['rate']
        
        print(f"[{i:3d}] {traffic:20s} rate={rate:.5f}...", end=' ', flush=True)
        
        # Update config with traffic pattern (no hotspot node needed)
        update_config(config_file, traffic, rate)
        
        output_file = f'sim{i}.stats'
        start_time = time.time()
        success = run_simulation(config_file, output_file)
        duration = time.time() - start_time
        
        metrics = extract_metrics(output_file)
        
        if metrics and 'avg_latency' in metrics:
            # Store raw metrics WITHOUT pre-defined labels
            data.append({
                'step': i,
                'traffic_pattern': traffic,
                'injection_rate': round(rate, 6),
                'network_load': round(metrics.get('network_load', 0), 6),
                'throughput': round(metrics.get('throughput', 0), 6),
                'avg_latency': round(metrics.get('avg_latency', 0), 2),
                'network_latency': round(metrics.get('network_latency', 0), 2),
                'unstable': 1 if metrics.get('unstable', False) else 0,
                # No pre-defined hotspot labels - will be detected later
            })
            
            status = "✓" if success else "⚠"
            print(f"{status} ({duration:.1f}s) lat={metrics.get('avg_latency', 0):.1f}")
        else:
            print(f"✗ ({duration:.1f}s) - Failed")
        
        try:
            os.remove(output_file)
        except:
            pass
    
    return data

def main():
    config_file = os.path.join(os.path.dirname(__file__), 'booksim_hotspot.config')
    
    print("\n" + "="*70)
    print(" BookSim2 Dataset Generation (NATURAL HOTSPOT DETECTION)")
    print("="*70)
    print(" Configuration: 8x8 mesh, natural traffic patterns only")
    print(" NO MANUAL HOTSPOT FORCING - hotspots will emerge naturally")
    print(" Traffic patterns: uniform, transpose, shuffle, tornado, neighbor, bitcomp")
    print(" Varying injection rates to create natural congestion scenarios")
    print(" Will generate ~400 raw samples for natural hotspot detection")
    print("="*70)
    
    all_data = []
    
    # Scenario 1: Uniform Random Traffic - 80 samples
    # Low to high injection rates - some may create natural hotspots
    uniform_params = []
    for rate in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02]:
        for rep in range(7):
            uniform_params.append({
                'traffic': 'uniform',
                'rate': rate + random.uniform(-0.0005, 0.0005),
            })
    
    data1 = generate_scenario_samples(config_file, "Uniform Random Traffic - 80 samples",
                                      uniform_params[:80], start_idx=1)
    all_data.extend(data1)
    
    # Scenario 2: Transpose Traffic - 60 samples
    # Bit-reverse transpose can create congestion patterns
    transpose_params = []
    for rate in [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.012, 0.015, 0.018]:
        for rep in range(6):
            transpose_params.append({
                'traffic': 'transpose',
                'rate': rate + random.uniform(-0.0003, 0.0003),
            })
    
    data2 = generate_scenario_samples(config_file, "Transpose Traffic - 60 samples",
                                      transpose_params[:60], start_idx=len(all_data)+1)
    all_data.extend(data2)
    
    # Scenario 3: Shuffle Traffic - 60 samples
    # Perfect shuffle can create natural congestion
    shuffle_params = []
    for rate in [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.012, 0.015, 0.018]:
        for rep in range(6):
            shuffle_params.append({
                'traffic': 'shuffle',
                'rate': rate + random.uniform(-0.0003, 0.0003),
            })
    
    data3 = generate_scenario_samples(config_file, "Shuffle Traffic - 60 samples",
                                      shuffle_params[:60], start_idx=len(all_data)+1)
    all_data.extend(data3)
    
    # Scenario 4: Tornado Traffic - 60 samples
    # Tornado pattern can create severe congestion at certain nodes
    tornado_params = []
    for rate in [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.012, 0.015, 0.018]:
        for rep in range(6):
            tornado_params.append({
                'traffic': 'tornado',
                'rate': rate + random.uniform(-0.0003, 0.0003),
            })
    
    data4 = generate_scenario_samples(config_file, "Tornado Traffic - 60 samples",
                                      tornado_params[:60], start_idx=len(all_data)+1)
    all_data.extend(data4)
    
    # Scenario 5: Neighbor Traffic - 40 samples
    # Nearest neighbor communication can create localized congestion
    neighbor_params = []
    for rate in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
        for rep in range(5):
            neighbor_params.append({
                'traffic': 'neighbor',
                'rate': rate + random.uniform(-0.0002, 0.0002),
            })
    
    data5 = generate_scenario_samples(config_file, "Neighbor Traffic - 40 samples",
                                      neighbor_params[:40], start_idx=len(all_data)+1)
    all_data.extend(data5)
    
    # Scenario 6: BitComp Traffic - 40 samples
    # Bit complement can create specific congestion patterns
    bitcomp_params = []
    for rate in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
        for rep in range(5):
            bitcomp_params.append({
                'traffic': 'bitcomp',
                'rate': rate + random.uniform(-0.0002, 0.0002),
            })
    
    data6 = generate_scenario_samples(config_file, "BitComp Traffic - 40 samples",
                                      bitcomp_params[:40], start_idx=len(all_data)+1)
    all_data.extend(data6)
    
    if not all_data:
        print("\n✗ Error: No valid data collected!")
        return
    
    df = pd.DataFrame(all_data)
    
    # Add natural hotspot detection based on metrics
    df_with_labels = detect_natural_hotspots(df)
    
    output_csv = 'booksim_dataset_raw.csv'
    df_with_labels.to_csv(output_csv, index=False)
    
    print("\n" + "="*70)
    print(f" RAW DATASET READY: {output_csv}")
    print("="*70)
    print(f"\nTotal samples: {len(df_with_labels)}")
    print(f"\nTraffic Pattern Distribution:")
    print(df_with_labels['traffic_pattern'].value_counts())
    print(f"\nNatural Hotspot Detection Results:")
    hotspot_count = df_with_labels['hotspot_detected'].sum()
    print(f"  Samples with natural hotspots detected: {hotspot_count} ({hotspot_count/len(df_with_labels)*100:.1f}%)")
    print(f"  Normal samples: {len(df_with_labels) - hotspot_count} ({(len(df_with_labels) - hotspot_count)/len(df_with_labels)*100:.1f}%)")
    
    print(f"\nHotspots by Traffic Pattern:")
    hotspot_by_pattern = df_with_labels.groupby('traffic_pattern')['hotspot_detected'].agg(['sum', 'count'])
    hotspot_by_pattern['percentage'] = (hotspot_by_pattern['sum'] / hotspot_by_pattern['count'] * 100).round(1)
    print(hotspot_by_pattern)
    
    print(f"\nBasic Statistics:")
    print(df_with_labels[['injection_rate', 'network_load', 'throughput', 'avg_latency']].describe())
    
    print(f"\n✓ Raw dataset saved: {os.path.abspath(output_csv)}")
    print(f"  Columns: step, traffic_pattern, injection_rate, network_load,")
    print(f"            throughput, avg_latency, network_latency, unstable,")
    print(f"            hotspot_detected, congestion_score")
    print(f"  NATURAL HOTSPOT DETECTION: Based on latency and throughput analysis")
    print(f"  NO MANUAL HOTSPOT FORCING: All patterns are natural BookSim traffic")
    print(f"  Next step: Run train_lstm_model.py to train on this naturally labeled dataset")

def detect_natural_hotspots(df):
    """
    Detect natural hotspots based on network performance metrics.
    Uses statistical analysis to identify congestion patterns.
    """
    print(f"\n{'='*70}")
    print(" NATURAL HOTSPOT DETECTION")
    print(f"{'='*70}")
    
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
    
    print(f"Congestion score threshold: {congestion_threshold:.3f}")
    print(f"Latency threshold: {latency_threshold:.1f}")
    print(f"Hotspots detected: {df['hotspot_detected'].sum()} / {len(df)} samples")
    
    # Clean up temporary columns
    df = df.drop(['latency_score', 'throughput_score', 'load_efficiency', 'efficiency_score'], axis=1)
    
    return df

if __name__ == "__main__":
    main()
