# DeepMedico™ Health Sensing: Breathing Irregularity Detection During Sleep

## Project Overview

This repository contains the full pipeline for detecting breathing irregularities such as **Hypopnea** and **Obstructive Apnea** during sleep using physiological signals. The project is a part of DeepMedico™'s pilot study involving **5 overnight sleep recordings** (8 hours each) from different participants.

---
## Project Structure
deepmedico-health-sensing/
├── Data/ # Raw signals for each participant
│ └── AP01/ to AP05/
├── Cleaned/ # Filtered signals (bandpass)
├── Dataset/ # Final windowed dataset used for training
├── Visualizations/ # PDF plots of all signals + annotations
├── models/
│ ├── cnn_model.py # 1D CNN model training script
│ └── conv_lstm_model.py # 1D Conv-LSTM model training script
├── vis.py # Visualization script
├── clean_signals.py # Signal filtering pipeline
├── create_dataset.py # Dataset generation script
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## Problem Statement

Detect abnormal breathing patterns such as:
- **Hypopnea**: Shallow breathing
- **Obstructive Apnea**: Airway blockage causing stopped breathing

### Goal
Build models to classify 30-second windows of physiological signals into:
- `Normal`
- `Hypopnea`
- `Obstructive Apnea`
---
## Dataset Description

Each participant's folder (e.g., `AP01/`) contains:

| File Name             | Description                                  | Sampling Rate |
|----------------------|----------------------------------------------|---------------|
| `nasal_airflow.txt`  | Measures airflow through the nose            | 32 Hz         |
| `thoracic_movement.txt` | Chest expansion data                        | 32 Hz         |
| `spo2.txt`           | Blood oxygen saturation                      | 4 Hz          |
| `flow_events.txt`    | Annotated irregularities (hypopnea/apnea)   | Event-based   |
| `sleep_profile.txt`  | Sleep stages over time (optional)           | Event-based   |

---

## Methodology

### 1. Visualization (`vis.py`)
- Aligns multi-rate signals using timestamps
- Overlays annotated events
- Exports PDFs to `Visualizations/`

```bash
python vis.py -name "Data/AP01"

2. Signal Cleaning (clean_signals.py)
Applied Butterworth bandpass filter (0.17–0.4 Hz) to remove noise

Cleaned signals saved to Cleaned/

3. Dataset Creation (create_dataset.py)
30-second windows with 50% overlap

Labeled using flow events:

50% overlap with event → label as event

else → Normal

Output: Dataset/breathing_dataset.csv
```bash
python create_dataset.py -in_dir Data -out_dir Dataset

4. Modeling and Evaluation
A. 1D CNN (cnn_model.py)
Extracts temporal features using convolution layers

B. 1D Conv-LSTM (conv_lstm_model.py)
Combines CNN spatial encoding with LSTM temporal memory

Evaluation Strategy:
Leave-One-Participant-Out Cross-Validation (LOPO-CV)

```bash
python models/cnn_model.py
python models/conv_lstm_model.py
Results Summary
Model	Accuracy	Hypopnea Recall	Apnea Recall	Normal Recall
1D CNN	93%	0.0079	0.0000	0.9872
Conv-LSTM	95%	0.0000	0.0000	1.0000


Observation:
Excellent detection of Normal breathing

Poor recall for Hypopnea and Apnea due to class imbalance

Requirements
Add all dependencies to requirements.txt:

numpy
pandas
scipy
matplotlib
seaborn
tensorflow
scikit-learn

Install them via:

pip install -r requirements.txt






