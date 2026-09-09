# 🔬 Network-on-Chip Hotspot Detection & Prediction Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Coming%20Soon-blue.svg)]()

> A comprehensive framework for detecting, analyzing, and predicting congestion hotspots in Network-on-Chip (NoC) architectures using machine learning and temporal analysis.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [📊 Methodology](#-methodology)
- [🔧 Architecture](#-architecture)
- [📈 Results & Visualizations](#-results--visualizations)
- [📁 Project Structure](#-project-structure)
- [🔬 Technical Details](#-technical-details)
- [👥 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## 🎯 Overview

This project implements an advanced NoC hotspot detection and prediction system that combines:

- **Cycle-accurate simulation** using BookSim2
- **Natural congestion analysis** without manual hotspot forcing
- **External traffic trace support** for real-world data
- **Temporal persistence analysis** of congestion events
- **Node-level identification** of responsible components
- **Predictive machine learning** for proactive hotspot detection

The framework automatically adapts to different data sources while providing consistent, academically rigorous analysis suitable for research and evaluation.

## ✨ Key Features

### 🔍 Detection & Analysis
- ✅ **Natural Hotspot Detection**: Statistical analysis of network congestion
- ✅ **Severity Classification**: Mild/Moderate/Severe categorization
- ✅ **Temporal Persistence**: Duration analysis of congestion episodes
- ✅ **Node-Level Identification**: Responsible node detection (external traces)
- ✅ **Traffic Pattern Risk Ranking**: Comparative analysis across patterns

### 🤖 Machine Learning
- ✅ **Bidirectional LSTM**: Temporal sequence prediction
- ✅ **1-Step Ahead Prediction**: Proactive hotspot forecasting
- ✅ **Robust Training**: Early stopping and regularization

### 📊 Data Sources
- ✅ **BookSim2 Integration**: Automatic dataset generation
- ✅ **External Trace Support**: CSV/text files with clock cycles and node pairs
- ✅ **Dataset-Agnostic**: Unified pipeline for multiple input formats

### 🎨 Visualization
- ✅ **Congestion Evolution**: Time-series plots for BookSim data
- ✅ **Density Analysis**: Packet density visualization for traces
- ✅ **Clean Outputs**: Academic-quality plots for presentations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- BookSim2 (optional, for dataset generation)
- 2-3 minutes execution time

### Installation

```bash
# Clone repository
git clone https://github.com/sonamgupta01/noc-hotspot-detection.git
cd noc-hotspot-detection

# Create virtual environment
python3 -m venv noc_env
source noc_env/bin/activate  # Linux/macOS
# noc_env\Scripts\activate   # Windows

# Install dependencies
pip install -r src/requirements.txt

# Run complete pipeline
python src/main.py
```

### Supported Platforms
- ✅ Linux (Ubuntu, Debian, Fedora)
- ✅ macOS (Intel & Apple Silicon)
- ✅ Windows (WSL2 recommended)

## 📊 Methodology

### 1. Data Acquisition
**BookSim2 Simulation**:
- 8×8 mesh topology (64 nodes)
- Dimension-Ordered Routing (DOR)
- Natural traffic patterns: uniform, transpose, shuffle, tornado, neighbor, bitcomp

**External Traces**:
- Time-stamped packet data
- Source-destination node pairs
- Automatic time-window segmentation

### 2. Congestion Detection

**BookSim Data**:
```
Congestion Score = 0.4×Latency + 0.3×Throughput + 0.2×Efficiency + 0.1×Instability
```

**External Traces**:
```
Packet Density = Σ packets_per_node / total_packets
Hotspot = Top 20% density windows
```

### 3. Machine Learning Pipeline
- **Input**: 10-timestep sequences of normalized metrics
- **Model**: Bidirectional LSTM with dropout
- **Output**: Binary hotspot prediction (1-step ahead)
- **Training**: 100% accuracy achieved on validation set

## 🔧 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Loader   │───▶│   Analysis       │───▶│   Visualization │
│                 │    │   Engine         │    │                 │
│ • BookSim CSV   │    │ • Severity       │    │ • Time Series   │
│ • External TXT  │    │ • Persistence    │    │ • Density Plots │
│ • Auto-detect   │    │ • Node-level     │    │ • Academic      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │   LSTM Model     │
                   │ • Prediction     │
                   │ • Training       │
                   │ • Validation     │
                   └──────────────────┘
```

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `data_loader.py` | Input processing | Multi-format support, automatic detection |
| `main.py` | Pipeline orchestration | Separated analysis paths, visualization |
| `generate_raw_dataset.py` | BookSim integration | Natural traffic patterns, statistical analysis |
| `train_lstm_model.py` | ML training | Bidirectional LSTM, temporal prediction |

## 📈 Results & Visualizations

### BookSim Dataset Analysis
```
📈 Total Samples: 340
🔥 Hotspots Detected: 64 (18.8%)
⏱️ Average Persistence: 12.8 timesteps
🚦 High-Risk Patterns: BitComp, Transpose
📊 Visualization: congestion_evolution.png
```

### External Trace Analysis
```
📈 Total Time Windows: 538
🔥 Hotspots Detected: 100 (18.6%)
🔥 Unique Hotspot Nodes: 52
🆔 Top Nodes: 62(26), 0(22), 63(20)
📊 Visualization: density_evolution.png
```


### Phase-2: Hotspot Prediction & Adaptive Rerouting (Newly Added)

In Phase-2, the project is extended from hotspot detection to hotspot prediction and mitigation using a software-only approach.

**1. Continuous Hotspot Prediction (WHEN)**
A trained LSTM model is used to predict whether a hotspot will occur in the next time window.
The dataset is analyzed using fixed sliding windows of 10 timesteps (e.g., 1–10, 11–20, …).
For each window, the model outputs:
Hotspot probability
Classification: HOTSPOT / NO HOTSPOT
Severity level: Mild / Moderate / Severe
This enables early prediction, before congestion becomes critical.

**2. Hotspot Localization (WHERE)**
When a hotspot is predicted, the system identifies which nodes are likely to be congested.
Node identification is based on traffic pattern–aware analysis (e.g., transpose, shuffle, tornado, bit-complement).
This makes the prediction explainable and defensible, rather than random.

**3. Hotspot-Aware Adaptive Rerouting**
Rerouting is applied only for windows where a hotspot is detected.
A local, software-based rerouting policy is used:
Hotspot nodes are avoided
Traffic is redistributed to low-load neighboring nodes
Traffic redistribution strategy:
60% traffic rerouted
40% traffic kept on original path
Load is split across multiple neighbors to prevent secondary congestion.

**4. Post-Rerouting Congestion Verification**
After rerouting, congestion is re-evaluated.
A reduced congestion probability confirms that rerouting helped:
Status reported as CONGESTION REDUCED
This provides a closed-loop feedback mechanism.

**5. Summary & Monitoring**
The system prints:
Total windows analyzed
Number of hotspot and non-hotspot windows
Severity distribution
Hotspots by traffic pattern
This demonstrates real-time monitoring with intelligent decision-making.


### ✔ Phase-2 Outcome

Phase-2 successfully transforms the project from passive hotspot detection into an active, predictive, and corrective NoC traffic management framework, fully implemented in software without any hardware modification.




### Sample Visualizations

#### BookSim Congestion Evolution
![Congestion Evolution](booksim_congestion_evolution.png)

#### External Trace Density Analysis
![Density Evolution](external_density_evolution.png)

## 📁 Project Structure

```
noc-hotspot-detection/
├── src/
│   ├── main.py                    # Pipeline orchestrator
│   ├── data_loader.py             # Multi-format data handler
│   ├── generate_raw_dataset.py    # BookSim dataset creation
│   ├── train_lstm_model.py        # LSTM training module
│   ├── requirements.txt           # Python dependencies
│   └── booksim_hotspot.config     # BookSim configuration
├── booksim_dataset_raw.csv        # Generated dataset
├── lstm_hotspot_model.h5          # Trained model
├── lstm_training_history.png      # Training visualization
|── predict_hotspot.py             # prediction and rerouting
├── booksim_congestion_evolution.png # BookSim analysis plot
├── external_density_evolution.png  # External analysis plot
├── Temp1A.txt                     # Sample external trace
├── README.md                      # This file
└── Literature_Review_ReSubmission_Grp_37.docx
```

## 🔬 Technical Details

### Hotspot Detection Algorithms

**BookSim (Statistical)**:
- Multi-metric congestion scoring
- Percentile-based thresholding
- Traffic pattern analysis

**External Traces (Density-Based)**:
- Time-window segmentation
- Per-node packet counting
- Top-k selection for hotspots
- Node responsibility identification

### Machine Learning Specifications

| Parameter | Value |
|-----------|-------|
| Model | Bidirectional LSTM |
| Sequence Length | 10 timesteps |
| Hidden Units | 64 |
| Dropout Rate | 0.2 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Validation Accuracy | 100% |

### Performance Metrics

- **Execution Time**: 2-3 minutes
- **Memory Usage**: < 500MB
- **Accuracy**: 100% on validation
- **Scalability**: Supports various NoC sizes

## 👥 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BookSim2 Team** at Stanford University for the excellent simulator
- **TensorFlow/Keras** community for machine learning frameworks
- **Academic Advisors** for guidance and research support
- **Open Source Community** for invaluable tools and libraries

## 📞 Contact

**Sonam Gupta**
- 📧 Email: sonam98450@gmail.com
- 🐙 GitHub: [@sonamgupta01](https://github.com/sonamgupta01)
- 📱 LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

### Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/sonamgupta01/noc-hotspot-detection/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/sonamgupta01/noc-hotspot-detection/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/sonamgupta01/noc-hotspot-detection/wiki)

---

## 🎓 Academic Context

This framework addresses key research gaps in NoC congestion analysis:

- **Natural Detection**: Avoids artificial hotspot injection
- **Temporal Analysis**: Captures congestion dynamics over time
- **Node Attribution**: Identifies responsible network components
- **Predictive Capability**: Enables proactive congestion management
- **Multi-Source Support**: Bridges simulation and real-trace analysis

**Research Applications**:
- NoC architecture optimization
- Congestion-aware routing algorithms
- Machine learning for network monitoring
- Chip multiprocessor performance analysis

---

### Running project 
**phase 1**
source noc_env/bin/activate
python src/main.py
**phase 2**
open other terminal
source noc_env/bin/activate
python predict_hotspot.py

open other terminal (To see the result after rerouting)
source noc_env/bin/activate
python external_hotspot.py


⭐ **If you find this project useful, please give it a star!**

*Built with ❤️ for advancing Network-on-Chip research and education*

**Last Updated**: December 2025


