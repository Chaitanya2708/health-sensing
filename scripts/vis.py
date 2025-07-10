import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta

def parse_signal(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines if ';' in line and not line.startswith("Signal")]
    timestamps, values = zip(*[line.split(";") for line in data_lines])
    timestamps = pd.to_datetime(timestamps, format="%d.%m.%Y %H:%M:%S,%f")
    values = pd.to_numeric(values)
    return pd.Series(values, index=timestamps)

def parse_flow_events(file_path):
    events = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                parts = line.strip().split(";")
                if len(parts) != 4:
                    raise ValueError("Expected 4 parts")
                time_range, duration, label, stage = parts
                start_str, end_str = time_range.split("-")
                start = pd.to_datetime(start_str, format="%d.%m.%Y %H:%M:%S,%f")
                end_time = pd.to_datetime(end_str, format="%H:%M:%S,%f").time()
                end = start.normalize() + pd.Timedelta(
                    hours=end_time.hour, minutes=end_time.minute,
                    seconds=end_time.second, microseconds=end_time.microsecond
                )
                if end < start:
                    end += pd.Timedelta(days=1)
                events.append((start, end, label))
            except Exception as e:
                print(f"[WARN] Could not parse event line: {line.strip()} | Error: {e}")
    return events

def parse_sleep_profile(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines if ';' in line and not line.startswith("Signal")]
    timestamps, labels = zip(*[line.split(";") for line in data_lines])
    timestamps = pd.to_datetime(timestamps, format="%d.%m.%Y %H:%M:%S,%f")
    return pd.Series(labels, index=timestamps)

def visualize_participant(participant_path, participant_id, use_sleep=False):
    try:
        nasal = parse_signal(os.path.join(participant_path, "nasal_airflow.txt"))
        thoracic = parse_signal(os.path.join(participant_path, "thoracic_movement.txt"))
        spo2 = parse_signal(os.path.join(participant_path, "spo2.txt"))
        events = parse_flow_events(os.path.join(participant_path, "flow_events.txt"))
        sleep_profile = parse_sleep_profile(os.path.join(participant_path, "sleep_profile.txt")) if use_sleep else None
    except Exception as e:
        print(f"[ERROR] Failed to parse signals for {participant_id}: {e}")
        return

    os.makedirs("Visualizations", exist_ok=True)
    pdf_path = os.path.join("Visualizations", f"{participant_id}.pdf")

    with PdfPages(pdf_path) as pdf:
        start_time = nasal.index.min()
        end_time = nasal.index.max()

        current_time = start_time
        while current_time < end_time:
            window_end = current_time + timedelta(minutes=5)

            fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            fig.suptitle(f"{participant_id} | {current_time:%H:%M:%S} to {window_end:%H:%M:%S}", fontsize=14)

            axs[0].plot(nasal[current_time:window_end], label="Nasal Airflow", color='blue')
            axs[0].set_ylabel("Nasal")

            axs[1].plot(thoracic[current_time:window_end], label="Thoracic Movement", color='orange')
            axs[1].set_ylabel("Thoracic")

            axs[2].plot(spo2[current_time:window_end], label="SpO₂", color='green')
            axs[2].set_ylabel("SpO₂ (%)")
            axs[2].set_xlabel("Time")

            # Event overlays
            for start, end, label in events:
                if start <= window_end and end >= current_time:
                    for ax in axs:
                        ax.axvspan(start, end, color='red' if 'Apnea' in label else 'purple', alpha=0.3)
                        ax.text((start + (end - start) / 2), ax.get_ylim()[1]*0.9, label,
                                color='black', ha='center', va='top', fontsize=8)

            # Sleep stage annotations (on SpO2)
            if sleep_profile is not None:
                sp = sleep_profile[current_time:window_end]
                for time, stage in sp.items():
                    axs[2].text(time, axs[2].get_ylim()[0], stage, fontsize=6,
                                color='gray', rotation=90, ha='center')

            # X-axis ticks every 5 seconds, limited count
            axs[-1].xaxis.set_major_locator(mdates.SecondLocator(interval=5))
            axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=20))

            for ax in axs:
                ax.grid(True, linestyle='--', linewidth=0.5)

            plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()
            current_time = window_end

    print(f"[✓] Saved PDF: {pdf_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, required=True, help="Path to Data directory (e.g., Data)")
    parser.add_argument("--sleep", action='store_true', help="Overlay sleep profile if available")
    args = parser.parse_args()

    data_dir = args.name
    participants = sorted([p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))])
    for pid in participants:
        print(f"Visualizing {pid}...")
        visualize_participant(os.path.join(data_dir, pid), pid, use_sleep=args.sleep)

if __name__ == "__main__":
    main()
