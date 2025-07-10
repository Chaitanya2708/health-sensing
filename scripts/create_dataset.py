import os
import pandas as pd
import numpy as np
from datetime import timedelta

# Parameters
WINDOW_SIZE = timedelta(seconds=30)
STEP_SIZE = timedelta(seconds=15)
EVENT_LABELS = ['Hypopnea', 'Obstructive Apnea']

# Load a signal from cleaned CSV
def load_signal(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df.squeeze()  # Convert DataFrame to Series

# Load and parse flow event annotations
def load_events(events_path):
    events = []
    with open(events_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' not in line:
                continue
            parts = line.strip().split(";")
            if len(parts) != 4:
                continue
            time_range, _, label, _ = parts
            if label not in EVENT_LABELS:
                continue
            try:
                start_str, end_str = time_range.split("-")
                start = pd.to_datetime(start_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
                end = pd.to_datetime(end_str.strip(), format="%H:%M:%S,%f")

                # Reconstruct full end timestamp if date missing
                end = start.normalize() + timedelta(
                    hours=end.hour, minutes=end.minute, seconds=end.second, microseconds=end.microsecond
                )
                if end < start:
                    end += timedelta(days=1)
                events.append((start, end, label))
            except Exception as e:
                continue
    return events

# Assign window label based on >50% overlap
def assign_label(window_start, window_end, events):
    window_duration = (window_end - window_start).total_seconds()
    for start, end, label in events:
        overlap_start = max(start, window_start)
        overlap_end = min(end, window_end)
        overlap = (overlap_end - overlap_start).total_seconds()
        if overlap > 0 and (overlap / window_duration) > 0.5:
            return label
    return 'Normal'

# Main dataset creation logic
def create_dataset(data_dir='Cleaned', event_dir='Data', output_csv='Dataset/breathing_dataset.csv'):
    os.makedirs('Dataset', exist_ok=True)
    rows = []

    for participant in sorted(os.listdir(data_dir)):
        print(f"Processing {participant}...")
        part_path = os.path.join(data_dir, participant)
        try:
            nasal = load_signal(os.path.join(part_path, 'nasal_airflow_cleaned.csv'))
            thoracic = load_signal(os.path.join(part_path, 'thoracic_movement_cleaned.csv'))
            spo2 = load_signal(os.path.join(part_path, 'spo2_cleaned.csv'))

            events = load_events(os.path.join(event_dir, participant, 'flow_events.txt'))

            start_time = max(nasal.index[0], thoracic.index[0], spo2.index[0])
            end_time = min(nasal.index[-1], thoracic.index[-1], spo2.index[-1])

            current_time = start_time
            while current_time + WINDOW_SIZE <= end_time:
                w_start = current_time
                w_end = current_time + WINDOW_SIZE
                label = assign_label(w_start, w_end, events)

                nasal_window = nasal[w_start:w_end].values
                thoracic_window = thoracic[w_start:w_end].values
                spo2_window = spo2[w_start:w_end].values

                if len(nasal_window) == 0 or len(thoracic_window) == 0 or len(spo2_window) == 0:
                    current_time += STEP_SIZE
                    continue

                row = {
                    'participant': participant,
                    'start_time': w_start,
                    'end_time': w_end,
                    'label': label,
                    'nasal': nasal_window.tolist(),
                    'thoracic': thoracic_window.tolist(),
                    'spo2': spo2_window.tolist()
                }
                rows.append(row)
                current_time += STEP_SIZE
        except Exception as e:
            print(f"[ERROR] Failed for {participant}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved dataset to {output_csv}")

if __name__ == '__main__':
    create_dataset()
