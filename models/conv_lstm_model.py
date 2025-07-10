import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

LABELS = ['Normal', 'Hypopnea', 'Obstructive Apnea']
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}

# ------------------- Preprocessing -------------------

def preprocess_features(df, features=['nasal', 'thoracic', 'spo2']):
    X = []
    expected_length = 960  # 30 seconds * 32 Hz
    skipped = 0
    for _, row in df.iterrows():
        combined = []
        for feat in features:
            signal = np.array(eval(row[feat]))
            if len(signal) < expected_length:
                signal = np.pad(signal, (0, expected_length - len(signal)), mode='constant')
            elif len(signal) > expected_length:
                signal = signal[:expected_length]
            combined.append(signal)
        try:
            combined = np.stack(combined, axis=-1)  # (timesteps, channels)
            X.append(combined)
        except Exception:
            skipped += 1
    print(f"[INFO] Skipped {skipped} invalid windows.")
    X = np.array(X)
    X = X.reshape((X.shape[0], 6, 160, X.shape[2]))  # reshape to (samples, timesteps, steps_per_timestep, channels)
    return X

def preprocess_labels(df):
    y = df['label'].map(LABEL2IDX)
    return to_categorical(y, num_classes=len(LABELS))

# ------------------- Model -------------------

def build_conv_lstm(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(Conv1D(32, kernel_size=5, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Conv1D(64, kernel_size=5, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------- Evaluation -------------------

def evaluate_model(y_true, y_pred, labels=LABELS):
    print(classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    metrics = defaultdict(list)
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        metrics[label].append({
            "accuracy": accuracy_score(y_true == i, y_pred == i),
            "precision": precision_score(y_true, y_pred, average=None, zero_division=0)[i],
            "recall": recall_score(y_true, y_pred, average=None, zero_division=0)[i],
            "sensitivity": TP / (TP + FN) if (TP + FN) > 0 else 0,
            "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
            "confusion_row": cm[i].tolist()
        })
    return metrics

# ------------------- Main -------------------

def main():
    df = pd.read_csv("Dataset/breathing_dataset.csv")
    X = preprocess_features(df)
    y_cat = preprocess_labels(df)
    y = np.argmax(y_cat, axis=1)
    groups = df['participant'].values

    logo = LeaveOneGroupOut()
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_cat, groups)):
        print(f"\n[INFO] Fold: Leave out {groups[test_idx[0]]}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]
        y_test_labels = y[test_idx]

        model = build_conv_lstm(X_train.shape[1:], y_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        metrics = evaluate_model(y_test_labels, y_pred)
        fold_metrics.append(metrics)

    print("\n[INFO] Aggregated Metrics")
    for label in LABELS:
        acc = [fold[label][0]['accuracy'] for fold in fold_metrics]
        prec = [fold[label][0]['precision'] for fold in fold_metrics]
        rec = [fold[label][0]['recall'] for fold in fold_metrics]
        sens = [fold[label][0]['sensitivity'] for fold in fold_metrics]
        spec = [fold[label][0]['specificity'] for fold in fold_metrics]

        print(f"\nLabel: {label}")
        print(f"  Accuracy:    {np.mean(acc):.4f} ± {np.std(acc):.4f}")
        print(f"  Precision:   {np.mean(prec):.4f} ± {np.std(prec):.4f}")
        print(f"  Recall:      {np.mean(rec):.4f} ± {np.std(rec):.4f}")
        print(f"  Sensitivity: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
        print(f"  Specificity: {np.mean(spec):.4f} ± {np.std(spec):.4f}")

if __name__ == '__main__':
    main()
