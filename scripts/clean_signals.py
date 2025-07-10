import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def load_signal(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = [line.strip() for line in lines if ';' in line and not line.startswith("Start")]
    timestamps, values = zip(*[line.split(";") for line in data])
    timestamps = pd.to_datetime(timestamps, format="%d.%m.%Y %H:%M:%S,%f")
    values = pd.to_numeric(values)
    return pd.Series(values, index=timestamps)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

def clean_signal(signal_series, fs, kind="bandpass"):
    if kind == "bandpass":
        # For nasal and thoracic (breathing range 0.17–0.4 Hz)
        cleaned = butter_bandpass_filter(signal_series.values, 0.17, 0.4, fs)
    elif kind == "lowpass":
        # For SpO₂ (slow trend, cutoff below 0.1 Hz)
        cleaned = butter_lowpass_filter(signal_series.values, cutoff=0.1, fs=fs)
    else:
        raise ValueError("Invalid filter type.")
    return pd.Series(cleaned, index=signal_series.index)

def process_participant(participant_path, output_path):
    nasal_path = os.path.join(participant_path, 'nasal_airflow.txt')
    thoracic_path = os.path.join(participant_path, 'thoracic_movement.txt')
    spo2_path = os.path.join(participant_path, 'spo2.txt')

    nasal = load_signal(nasal_path)
    thoracic = load_signal(thoracic_path)
    spo2 = load_signal(spo2_path)

    # Assume all signals sampled at 32 Hz
    nasal_cleaned = clean_signal(nasal, fs=32, kind="bandpass")
    thoracic_cleaned = clean_signal(thoracic, fs=32, kind="bandpass")
    spo2_cleaned = clean_signal(spo2, fs=32, kind="lowpass")

    os.makedirs(output_path, exist_ok=True)
    nasal_cleaned.to_csv(os.path.join(output_path, 'nasal_airflow_cleaned.csv'))
    thoracic_cleaned.to_csv(os.path.join(output_path, 'thoracic_movement_cleaned.csv'))
    spo2_cleaned.to_csv(os.path.join(output_path, 'spo2_cleaned.csv'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', type=str, required=True, help="Input raw Data directory (e.g., 'Data')")
    parser.add_argument('-out_dir', type=str, required=True, help="Output Cleaned directory (e.g., 'Cleaned')")
    args = parser.parse_args()

    input_dir = args.in_dir
    output_dir = args.out_dir

    for participant in sorted(os.listdir(input_dir)):
        in_path = os.path.join(input_dir, participant)
        out_path = os.path.join(output_dir, participant)
        if os.path.isdir(in_path):
            print(f"[INFO] Cleaning signals for {participant}...")
            try:
                process_participant(in_path, out_path)
                print(f"[✓] Cleaned data saved to {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed to clean {participant}: {e}")

if __name__ == '__main__':
    main()
