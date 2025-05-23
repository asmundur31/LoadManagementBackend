from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import joblib
import os
import numpy as np

def build_model(model_name: str = "MiniRocket"):
    """
    Builds a model pipeline based on the specified model name.

    Args:
        model_name (str): Name of the model to use ("MiniRocket", etc.)

    Returns:
        model: scikit-learn compatible pipeline
    """
    if model_name == "MiniRocket":
        model = Pipeline([
            ("minirocket", MiniRocket(num_kernels=5000, n_jobs=1)),
            ("classifier", LogisticRegression(max_iter=1000, verbose=1, random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_model(model, X_train, y_train, save_path=None):
    """
    Trains or loads a model and prints evaluation metrics.

    Args:
        model: scikit-learn compatible pipeline
        X_train: training data (numpy array or DataFrame)
        y_train: training labels
        X_test: test data (numpy array or DataFrame)
        y_test: test labels
        save_path: optional path to save/load the model

    Returns:
        trained model
    """
    if save_path and os.path.exists(save_path):
        print(f"\t📦 Loading model from {save_path}")
        model = joblib.load(save_path)
    else:
        print("\t🚀 Training model...")
        model.fit(X_train, y_train)
        if save_path:
            joblib.dump(model, save_path)
            print(f"\t💾 Model saved to {save_path}")    
    return model


def predict(model, X_test):
    """
    Makes predictions using the trained model.

    Args:
        model: trained model
        X_test: test data (numpy array or DataFrame)

    Returns:
        y_pred: predicted labels
    """
    y_pred = model.predict(X_test)
    return y_pred

def timestamp_predictions(y_pred, y_test_full_len):
    window_size = 100 
    step_size = 25
    # Estimate where each window starts (assuming uniform spacing)
    window_starts = np.arange(0, len(y_pred)) * step_size

    # Dictionary to accumulate votes for each timestamp
    label_votes = defaultdict(list)

    for pred, start in zip(y_pred, window_starts):
        start = int(start)
        end = min(start + window_size, y_test_full_len)
        for i in range(start, end):
            label_votes[i].append(pred)

    # Apply majority rule: label = 1 if ≥ 3/4 votes are 1, else 0
    timestamp_preds = np.zeros(y_test_full_len, dtype=int)
    for i, votes in label_votes.items():
        if votes.count(1) >= 2:
            timestamp_preds[i] = 1
        else:
            timestamp_preds[i] = 0

    return timestamp_preds


def evaluate_model(y_test, y_pred):
    """
    Evaluates the model's performance using classification metrics.

    Args:
        y_test: true labels
        y_pred: predicted labels
    """
    report = classification_report(y_test, y_pred)
    print("📊 Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return report

def count(labels, min_jump_length=10):
    count = 0
    consecutive_ones = 0
    
    for i in range(1, len(labels)):
        if labels[i] == 1:
            consecutive_ones += 1
        else:
            if consecutive_ones >= min_jump_length:
                count += 1
            consecutive_ones = 0

    if consecutive_ones >= min_jump_length:
        count += 1

    return count

def count_jumps(y_test, y_pred):
    """
    Counts the number of jumps in the predicted labels.

    Args:
        y_test: true labels
        y_pred: predicted labels
    """
    predicted_jumps = count(y_pred)
    true_jumps = count(y_test)

    print(f"Number of jumps predicted by the model: {predicted_jumps}")
    print(f"Number of jumps in the true labels: {true_jumps}")
    return true_jumps, predicted_jumps

def get_jump_ranges(label_array):
    """
    Finds consecutive jump intervals (where label == 1)
    Returns a list of tuples: (start_index, end_index)
    """
    from itertools import groupby
    from operator import itemgetter

    ones = [i for i, val in enumerate(label_array) if val == 1]
    if not ones:
        return []

    # Group consecutive indices
    ranges = []
    for k, g in groupby(enumerate(ones), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
    return ranges

def calculate_coverage(true_labels, pred_labels):
    true_jumps = get_jump_ranges(true_labels)
    pred_jumps = get_jump_ranges(pred_labels)

    coverages = []
    prediction_excesses = []
    false_negatives = []
    false_positives = []

    # Compute coverage per true jump
    for t_start, t_end in true_jumps:
        true_duration = t_end - t_start + 1
        total_overlap = 0

        for p_start, p_end in pred_jumps:
            overlap_start = max(t_start, p_start)
            overlap_end = min(t_end, p_end)

            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start + 1
                total_overlap += overlap

        if total_overlap == 0:
            false_negatives.append((t_start, t_end))

        coverage = total_overlap / true_duration
        coverages.append(coverage)

    # Compute excess per predicted jump
    for p_start, p_end in pred_jumps:
        pred_duration = p_end - p_start + 1
        total_overlap = 0

        for t_start, t_end in true_jumps:
            overlap_start = max(p_start, t_start)
            overlap_end = min(p_end, t_end)

            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start + 1
                total_overlap += overlap

        excess_duration = pred_duration - total_overlap
        excess_ratio = excess_duration / pred_duration if pred_duration > 0 else 0
        prediction_excesses.append(excess_ratio)

        if total_overlap == 0:
            false_positives.append((p_start, p_end))
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0
    avg_excess = sum(prediction_excesses) / len(prediction_excesses) if prediction_excesses else 0

    return {
        "coverages": coverages,
        "avg_coverage": avg_coverage,
        "prediction_excesses": prediction_excesses,
        "avg_excess": avg_excess,
        "false_negatives": false_negatives,
        "false_positives": false_positives
    }


def plot_predictions(imu_data, y_test, y_pred):
    """
    Interactive plot of IMU data with true and predicted labels. 
    Navigate using arrow keys: ←/→ to move start index, ↑/↓ to change window size.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    start_index = 0
    size = 1040

    acc_columns = ['lower_back_accx', 'lower_back_accy', 'lower_back_accz']
    global_min = imu_data[acc_columns].min().min()
    global_max = imu_data[acc_columns].max().max()

    def update_plot():
        ax.clear()

        end_index = min(start_index + size, len(imu_data))
        imu_slice = imu_data.iloc[start_index:end_index]
        y_test_slice = y_test[start_index:end_index]
        y_pred_slice = y_pred[start_index:end_index]

        ax.plot(imu_slice['lower_back_accx'].values, label="Acc X", color='r')
        ax.plot(imu_slice['lower_back_accy'].values, label="Acc Y", color='g')
        ax.plot(imu_slice['lower_back_accz'].values, label="Acc Z", color='b')

        height = global_max

        # True labels – yellow bars
        ax.fill_between(
            range(len(y_test_slice)),
            0,
            -y_test_slice * height,
            color='gold',
            alpha=0.4,
            label="True Jumps"
        )

        # Predicted labels – purple bars with more opacity
        ax.fill_between(
            range(len(y_pred_slice)),
            0,
            y_pred_slice * height,
            color='mediumpurple',
            alpha=0.4,
            label="Predicted Jumps"
        )
        ax.set_title(f"IMU Data [{start_index}:{end_index}]")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Acceleration")
        ax.set_ylim(global_min, global_max)  # Fixed y-axis scale
        ax.legend()
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal start_index, size
        if event.key == 'right':
            start_index = min(start_index + size, len(imu_data) - size)
        elif event.key == 'left':
            start_index = max(start_index - size, 0)
        elif event.key == 'up':
            size = min(size + 100, len(imu_data) - start_index)
        elif event.key == 'down':
            size = max(100, size - 100)
        update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    plt.tight_layout()
    plt.show()


def scatter_plot_predictions(summary_df):
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(summary_df["true_jumps"], summary_df["predicted_jumps"], color='blue', label='Subjects')
    plt.plot([summary_df["true_jumps"].min(), summary_df["true_jumps"].max()],
            [summary_df["true_jumps"].min(), summary_df["true_jumps"].max()],
            color='red', linestyle='--', label='Perfect prediction')

    # Add labels like S1, S2, ...
    for i, row in summary_df.iterrows():
        subject_label = f"S{i+1}"  # Remap: S1 for first subject in summary_df
        plt.text(row["true_jumps"] + 0.1, row["predicted_jumps"], subject_label, fontsize=9)

    # Set integer ticks
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    # Labels and styling
    plt.xlabel("True jump count")
    plt.ylabel("Predicted jump count")
    plt.title("Predicted vs True jump count per subject")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()