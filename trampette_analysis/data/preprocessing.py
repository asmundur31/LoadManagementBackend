import pandas as pd
import numpy as np
from collections import Counter


def segment_training_data(imu_df: pd.DataFrame):
    window_size = 100 

    y_raw = imu_df["label"].values
    X_raw = imu_df.drop(columns=["timestamp", "label"]).values

    segments_X = []
    segments_y = []

    i = 0
    while i + window_size <= len(imu_df):
        window_labels = y_raw[i:i + window_size]
        unique_labels = np.unique(window_labels)

        # Only include window if all labels are the same
        if len(unique_labels) == 1:
            segments_X.append(X_raw[i:i + window_size].T)  # (channels, time)
            segments_y.append(unique_labels[0])
            i += window_size  # Non-overlapping step
        else:
            i += 1  # Allow overlapping only around transitions

    return np.array(segments_X), np.array(segments_y)

def segment_test_data(imu_df: pd.DataFrame):
    window_size = 100
    stride = 25

    if "label" in imu_df.columns:
        X_raw = imu_df.drop(columns=["timestamp", "label"]).values
    else:
        X_raw = imu_df.drop(columns=["timestamp"]).values

    segments_X = []

    i = 0
    while i + window_size <= len(imu_df):
        segments_X.append(X_raw[i:i + window_size].T)

        i += stride

    return np.array(segments_X)