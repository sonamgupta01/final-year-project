#!/usr/bin/env python3
"""
LSTM Model for Temporal Hotspot Prediction in NoC.
Learns temporal patterns of network congestion and predicts future hotspots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import AUC, Precision, Recall
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Install with: pip3 install tensorflow")

def load_dataset(filename='booksim_dataset_large.csv'):
    """Load and prepare dataset."""
    print("="*70)
    print(" Loading Temporal Dataset")
    print("="*70)
    
    df = pd.read_csv(filename)
    print(f"\n✓ Loaded {len(df)} timesteps from {filename}")
    print(f"✓ Columns: {', '.join(df.columns)}")
    print(f"\nDataset temporal information:")
    print(f"  Total steps: {len(df)}")
    print(f"  Traffic patterns: {df['traffic_pattern'].unique().tolist()}")
    # Check if hotspot_node column exists (for backward compatibility)
    if 'hotspot_node' in df.columns:
        print(f"  Hotspot nodes: {df['hotspot_node'].unique().tolist()}")
    else:
        print(f"  Natural hotspot detection: {df['hotspot_detected'].sum()} hotspots detected")
    
    return df

def create_sequences(df, sequence_length=10, predict_horizon=1):
    """
    Create sequences for LSTM training.
    
    Args:
        df: DataFrame with temporal data (must be sorted by step)
        sequence_length: Number of timesteps per sequence
        predict_horizon: How many steps ahead to predict
    
    Returns:
        X: Input sequences [samples, timesteps, features]
        y: Target labels [samples]
        feature_names: List of feature names
    """
    print("\n" + "="*70)
    print(" Creating Temporal Sequences for LSTM")
    print("="*70)
    
    # Sort by traffic pattern and step to ensure temporal coherence
    df_sorted = df.sort_values(['traffic_pattern', 'step']).reset_index(drop=True)
    
    # Select features for LSTM
    feature_cols = [
        'injection_rate', 'network_load', 'throughput',
        'avg_latency', 'network_latency', 'unstable'
    ]
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = df_sorted.copy()
    df_normalized[feature_cols] = scaler.fit_transform(df_sorted[feature_cols])
    
    print(f"\n✓ Selected features: {feature_cols}")
    print(f"✓ Normalized using MinMaxScaler (0-1 range)")
    print(f"✓ Sequence length: {sequence_length} timesteps")
    print(f"✓ Prediction horizon: {predict_horizon} step(s) ahead")
    
    X = []
    y = []
    
    # Create sequences group by traffic pattern (maintain temporal coherence)
    for traffic_pattern in df_sorted['traffic_pattern'].unique():
        df_pattern = df_normalized[df_normalized['traffic_pattern'] == traffic_pattern].reset_index(drop=True)
        
        if len(df_pattern) < sequence_length + predict_horizon:
            print(f"  ⚠ Skipping '{traffic_pattern}' (too few samples: {len(df_pattern)})")
            continue
        
        # Create sliding window sequences
        for i in range(len(df_pattern) - sequence_length - predict_horizon + 1):
            seq = df_pattern.iloc[i:i+sequence_length][feature_cols].values
            target_idx = i + sequence_length + predict_horizon - 1
            target = df_pattern.iloc[target_idx]['hotspot_detected']
            
            X.append(seq)
            y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✓ Created {len(X)} sequences")
    print(f"  Shape: {X.shape} (samples, timesteps, features)")
    print(f"✓ Target distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y) * 100
        print(f"    Class {int(label)}: {count:4d} samples ({pct:5.1f}%)")
    
    return X, y, feature_cols, scaler

def build_lstm_model(sequence_length, num_features, lstm_units=64, dropout_rate=0.2):
    """
    Build LSTM model for temporal hotspot prediction.
    
    Architecture:
    - Input: Sequences of network metrics
    - Bidirectional LSTM layers (capture patterns in both directions)
    - Dropout for regularization
    - Dense output for binary classification
    """
    print("\n" + "="*70)
    print(" Building LSTM Model Architecture")
    print("="*70)
    
    model = Sequential([
        # Input layer implicitly defined by input_shape in first layer
        Bidirectional(LSTM(lstm_units, return_sequences=True, activation='relu'),
                     input_shape=(sequence_length, num_features)),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        Bidirectional(LSTM(lstm_units // 2, return_sequences=False, activation='relu')),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        
        # Output layer
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    print(f"\nModel Architecture:")
    print(f"  - Input: ({sequence_length}, {num_features})")
    print(f"  - Bidirectional LSTM: {lstm_units} units")
    print(f"  - Bidirectional LSTM: {lstm_units//2} units")
    print(f"  - Dense: 32 → 16 units")
    print(f"  - Output: 1 unit (sigmoid)")
    print(f"  - Dropout: {dropout_rate}")
    
    return model

def compile_and_display_model(model):
    """Compile model and display summary."""
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    print("\n" + "="*70)
    print(" Model Summary")
    print("="*70)
    model.summary()
    
    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    """Train LSTM model with early stopping."""
    print("\n" + "="*70)
    print(" Training LSTM Model")
    print("="*70)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,
                     verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                         min_lr=0.00001, verbose=1)
    ]
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Early stopping: patience=10 epochs")
    print(f"  LR reduction: factor=0.5, patience=5 epochs")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    return history

def evaluate_lstm_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "="*70)
    print(" Model Evaluation on Test Set")
    print("="*70)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Compute metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, confusion_matrix)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    print(f"\nDetailed Classification:")
    print(f"  True Negatives (Correct non-hotspot): {cm[0,0]}")
    print(f"  False Positives (Missed non-hotspot): {cm[0,1]}")
    print(f"  False Negatives (Missed hotspot):     {cm[1,0]}")
    print(f"  True Positives (Correct hotspot):     {cm[1,1]}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def plot_training_history(history, output_file='lstm_training_history.png'):
    """Plot training and validation metrics."""
    print("\n" + "="*70)
    print(" Plotting Training History")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Model Loss Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Model Accuracy Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Model Precision Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Model Recall Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history saved: {output_file}")
    plt.close()

def save_model(model, model_file='lstm_hotspot_model.h5'):
    """Save trained model."""
    model.save(model_file)
    print(f"✓ Model saved: {model_file}")

def main():
    """Main LSTM training pipeline."""
    if not TF_AVAILABLE:
        print("\n✗ ERROR: TensorFlow is required for LSTM training")
        print("Install with: pip3 install tensorflow")
        return
    
    print("\n" + "="*70)
    print(" LSTM Temporal Hotspot Prediction Model")
    print("="*70)
    print("\nObjective: Learn temporal patterns of network congestion")
    print("Task: Predict future hotspots based on historical traffic metrics")
    print()
    
    # Load dataset
    df = load_dataset('booksim_dataset_raw.csv')
    
    # Create sequences
    X, y, features, scaler = create_sequences(df, sequence_length=10, predict_horizon=1)
    
    if len(X) == 0:
        print("\n✗ Error: No sequences created. Check dataset.")
        return
    
    # Split data (70% train, 15% val, 15% test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 15% of 85%
    )
    
    print("\n" + "="*70)
    print(" Data Split")
    print("="*70)
    print(f"\nTrain set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Build model
    model = build_lstm_model(
        sequence_length=X.shape[1],
        num_features=X.shape[2],
        lstm_units=64,
        dropout_rate=0.2
    )
    
    # Compile
    model = compile_and_display_model(model)
    
    # Train
    history = train_lstm_model(
        model, X_train, y_train, X_val, y_val,
        epochs=50, batch_size=16
    )
    
    # Evaluate
    metrics = evaluate_lstm_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*70)
    print(" Training Complete!")
    print("="*70)
    print("\n✓ LSTM model trained successfully")
    print("✓ Model saved as: lstm_hotspot_model.h5")
    print("✓ Training history saved as: lstm_training_history.png")
    print("\nModel Capabilities:")
    print("  - Learns temporal dependencies in network traffic")
    print("  - Predicts future congestion patterns (hotspots)")
    print("  - Handles sequence data from NoC simulations")
    print("  - Bidirectional processing for improved context")
    print()

if __name__ == "__main__":
    main()
