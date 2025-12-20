# Network-on-Chip Hotspot Detection using Predictive Machine Learning

## 📋 Project Overview

This project implements a Network-on-Chip (NoC) hotspot detection and prediction framework using machine learning and temporal analysis.
It combines:
Cycle-accurate NoC simulation (BookSim2)
Natural hotspot detection using statistical congestion analysis
External traffic trace analysis
Predictive LSTM model for future hotspot prediction

The system is designed to detect when congestion happens, how severe it is, how long it persists, and which nodes are responsible, making it suitable for both simulation-based and real-trace-based evaluation.


### 📋 Requirements

- **Python 3.8+** with TensorFlow
- **BookSim2** (for data generation) or use pre-generated dataset
- **2-3 minutes** execution time

### 🖥️ Supported Platforms

✅ **Linux** (Ubuntu, Debian, Fedora, etc.)
✅ **macOS** (Intel & Apple Silicon)
✅ **Windows** (via WSL2 recommended)

📖 Methodology (Short & Clear)

1. NoC Simulation

Simulator: BookSim2

Topology: 8×8 mesh (64 nodes)

Routing: Dimension-Ordered Routing (DOR)

2. Dataset Generation

Natural traffic patterns (uniform, transpose, shuffle, tornado, etc.)

Hotspots identified using statistical congestion analysis

No manual hotspot forcing

3. Temporal Modeling

10-timestep sequences of network metrics

Features include latency, throughput, load, and stability

Min–max normalization

4. Predictive Learning

Bidirectional LSTM model

Binary hotspot prediction (1-step ahead)

Early stopping and dropout for robustness



### 🎯 Key Features (Literature Review Contributions)

✅ Natural hotspot detection (no manual forcing)
✅ Temporal hotspot persistence analysis
✅ Hotspot severity classification (Mild / Moderate / Severe)
✅ Traffic-pattern risk ranking
✅ Node-level hotspot identification (external traces)
✅ Predictive LSTM model (1-step ahead)
✅ Works on BookSim data + external datasets
---



🚀 How to Run (Single Command)

# Clone the repository
git clone https://github.com/sonamgupta01/noc-hotspot-detection.git
cd noc-hotspot-detection

# Create virtual environment (recommended)
python3 -m venv noc_env

# Activate virtual environment
source noc_env/bin/activate

# Install dependencies
pip install -r src/requirements.txt

# Run the complete pipeline
python src/main.py



📚 Technical Background
Network-on-Chip (NoC)

NoCs are scalable interconnects for multi-core processors. They route packets between cores (nodes) using switches and routers. Hotspot congestion occurs when many nodes send traffic to a single destination, causing:

    Increased latency
    Reduced throughput
    Network saturation
    Potential deadlock

BookSim2 Simulator

BookSim2 is a cycle-accurate interconnect network simulator developed at Stanford University. It accurately models:

    Router microarchitecture
    Buffer management
    Virtual channel allocation
    Flow control mechanisms
    Various traffic patterns




📊 What the Pipeline Produces
🔹 BookSim Dataset Results

Hotspot vs normal traffic analysis
Congestion score computation (latency + throughput + load)
Severity classification
Hotspot persistence (time-based)
Traffic pattern risk ranking

Visualization: congestion score vs timestep

🔹 External Dataset Results (e.g., Temp1A.txt)

Time-window–based hotspot detection
Packet density–based congestion analysis
Node-level hotspot identification using source/destination
Severity & persistence analysis
Visualization: packet density vs time




📂 Supported Input Data
1️⃣ BookSim Simulation Data

Generated internally using BookSim2 with traffic patterns:
uniform transpose shuffle tornado neighbor bitcomp

Metrics used: latency throughput network load instability flag

2️⃣ External Traffic Trace Data

Plain text or CSV with columns:
clock_cycle, source_node, destination_node

The same pipeline automatically adapts to this format.










🔥 Hotspot Detection Logic
BookSim Data

Hotspots are detected using a composite congestion score derived from:

high latency

low throughput

inefficient load utilization

instability

External Trace Data

Hotspots are detected using relative packet density spikes:

traffic is divided into time windows

top ~20% highest-density windows are marked as hotspots

node-level responsibility is identified using packet counts

🧠 Machine Learning Model

Model: Bidirectional LSTM

Input: Sequences of network metrics (10 timesteps)

Output: Hotspot prediction (binary)

Task: Predict hotspot before it occurs

Training: On BookSim-generated dataset
Artifacts:
lstm_hotspot_model.h5
lstm_training_history.png




📈 Visual Outputs
Dataset	Plot
BookSim	Congestion Score vs Timestep
External Trace	Packet Density vs Time Window

These plots help visualize when and how congestion evolves over time.

📁 Project Structure
noc-hotspot-detection/
├── src/
│   ├── main.py                    # Full pipeline runner
│   ├── generate_raw_dataset.py    # BookSim dataset generation
│   ├── train_lstm_model.py        # LSTM training
│   ├── data_loader.py             # BookSim + external data handler
│   └── requirements.txt
├── booksim_dataset_raw.csv
├── lstm_hotspot_model.h5
├── lstm_training_history.png
├── booksim_congestion_evolution.png
├── external_density_evolution.png
└── README.md


📚 Why This Project Is Different

Most NoC projects:

only detect congestion
only use simulators
do not analyze time behavior
do not identify responsible nodes

This project:
detects when
explains why
shows how severe
measures how long
identifies which nodes
predicts what happens next

🔬 Applicability

NoC research & experimentation
Chip multiprocessor congestion analysis
Trace-based traffic analysis
ML-based performance monitoring
Academic & research-level projects



👨‍💻 Author

Your Name

    GitHub: @sonamgupta01
    Email: sonam98450@gmail.com
    

🙏 Acknowledgments

    BookSim2 Team at Stanford University for the excellent simulator
    Open source community for machine learning libraries
    Academic advisors for guidance and support

📞 Contact & Support

For questions or issues:

    📧 Email: sonam98450@gmail.com
    🐛 Issues: GitHub Issues
    💬 Discussions: GitHub Discussions



⭐ Final Note

This repository contains a complete, modular, and extensible NoC hotspot analysis framework, validated using both simulated and external datasets, and enhanced with machine learning prediction.

⭐ Star This Repository

If you find this project useful, please give it a star! It helps others discover this work.

Stars

Built with ❤️ for Network-on-Chip Research
Last Updated: November 2025


