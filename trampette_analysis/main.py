import pandas as pd
import numpy as np
import os
import joblib
from config import config
from sklearn.preprocessing import StandardScaler
from data.loader import load_imu_data_and_annotations, combine_imu_data_and_annotations
from data.preprocessing import segment_training_data, segment_test_data 
from ml.models.jump_count_model import build_model, train_model, predict, evaluate_model, count_jumps, plot_predictions, timestamp_predictions, calculate_coverage, scatter_plot_predictions
from data.filteringAndNoise import butterworth_filter_high, butterworth_filter_low, downsample

acc_cols = ['lower_back_accx', 'lower_back_accy', 'lower_back_accz', 'chest_accx', 'chest_accy', 'chest_accz', 'thigh_accx', 'thigh_accy', 'thigh_accz', 'shin_accx', 'shin_accy', 'shin_accz']
gyro_cols = ['lower_back_gyrox', 'lower_back_gyroy', 'lower_back_gyroz', 'chest_gyrox', 'chest_gyroy', 'chest_gyroz', 'thigh_gyrox', 'thigh_gyroy', 'thigh_gyroz', 'shin_gyrox', 'shin_gyroy', 'shin_gyroz']
magn_cols = ['lower_back_magnx', 'lower_back_magny', 'lower_back_magnz', 'chest_magnx', 'chest_magny', 'chest_magnz', 'thigh_magnx', 'thigh_magny', 'thigh_magnz', 'shin_magnx', 'shin_magny', 'shin_magnz']

def main():
    subject_ids = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    results = []

    # Get the results
    summary_df = pd.read_csv("data_public/processed/loso_results_summary.csv")
    summary_df = summary_df.sort_values(by="test_subject").reset_index(drop=True)
    scatter_plot_predictions(summary_df)

    print("🔄 Starting LOSO cross-validation...")
    for test_id in subject_ids:
        print("🔄 Loading data...")
        train_ids = [sid for sid in subject_ids if sid != test_id]
        test_ids = [test_id]
        imu_df_train = []
        imu_df_test = []
        for id in train_ids:
            sensors_df, annotations_df, _ = load_imu_data_and_annotations(id)
            sensors_df = combine_imu_data_and_annotations(sensors_df, annotations_df)
            imu_df_train.append(sensors_df)
        for id in test_ids:
            sensors_df, annotations_df, _ = load_imu_data_and_annotations(id)
            sensors_df = combine_imu_data_and_annotations(sensors_df, annotations_df)
            imu_df_test.append(sensors_df)
        imu_df_train = pd.concat(imu_df_train, ignore_index=True)
        imu_df_test = pd.concat(imu_df_test, ignore_index=True)

        label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
        imu_df_test["label"] = imu_df_test["label"].map(label_map).values
        imu_df_train["label"] = imu_df_train["label"].map(label_map).values

        print("📂 Preprocessing data...")
        print("\t🏷️ Labeling and segmenting data...")
        # Filter data
        imu_df_train = butterworth_filter_low(imu_df_train, columns=acc_cols+gyro_cols+magn_cols, cutoff=20)
        imu_df_train = butterworth_filter_high(imu_df_train, columns=acc_cols+gyro_cols+magn_cols, cutoff=0.5)
        
        imu_df_test = butterworth_filter_low(imu_df_test, columns=acc_cols+gyro_cols+magn_cols, cutoff=20)
        imu_df_test = butterworth_filter_high(imu_df_test, columns=acc_cols+gyro_cols+magn_cols, cutoff=0.5)
        # Downsample data
        imu_df_train = downsample(imu_df_train, 2)
        imu_df_test = downsample(imu_df_test, 2)

        # Normalize data
        scaler = StandardScaler()
        imu_df_train_scaled = pd.DataFrame(
            scaler.fit_transform(imu_df_train[acc_cols + gyro_cols + magn_cols]),
            columns=acc_cols + gyro_cols + magn_cols
        )
        joblib.dump(scaler, 'data_public/models/final_jump_count_scaler.pkl')
        imu_df_test_scaled = pd.DataFrame(
            scaler.fit_transform(imu_df_test[acc_cols + gyro_cols + magn_cols]),
            columns=acc_cols + gyro_cols + magn_cols
        )
        imu_df_train_scaled["label"] = imu_df_train["label"].values
        imu_df_train_scaled["timestamp"] = imu_df_train["timestamp"].values
        imu_df_test_scaled["label"] = imu_df_test["label"].values
        imu_df_test_scaled["timestamp"] = imu_df_test["timestamp"].values
        # Segment data
        X_train, y_train = segment_training_data(imu_df_train_scaled)
        X_test = segment_test_data(imu_df_test_scaled)

        print("🧠 Build model...")
        model = build_model()
        model = train_model(model, X_train, y_train, save_path='data_public/models/final_jump_count_model.pkl')

        print("📈 Making predictions...")
        y_pred = predict(model, X_test)
        y_pred_timestamp = timestamp_predictions(y_pred, imu_df_test.shape[0])

        print("✅ Evaluating model...")
        report = evaluate_model(imu_df_test["label"], y_pred_timestamp)

        # Count jump events
        true_jump_count, predicted_jump_count = count_jumps(imu_df_test["label"], y_pred_timestamp)

        # Calculate coverage and excess
        coverage_results = calculate_coverage(imu_df_test["label"].values, y_pred_timestamp)
        avg_coverage = coverage_results["avg_coverage"]
        avg_excess = coverage_results["avg_excess"]
        false_negatives = len(coverage_results["false_negatives"])
        false_positives = len(coverage_results["false_positives"])
        true_positives = true_jump_count - false_negatives

        # Logging
        print(f"True jump count: {true_jump_count}")
        print(f"Predicted jump count: {predicted_jump_count}")
        print(f"True positives: {true_positives}")
        print(f"Average coverage of true jumps: {avg_coverage:.2f}")
        print(f"Average excess of predicted jumps: {avg_excess:.2f}")
        print(f"False negatives (missed true jumps): {false_negatives}")
        print(f"False positives (spurious predictions): {false_positives}")

        # Plotting
        print("📊 Plotting results...")
        plot_predictions(imu_df_test, imu_df_test["label"], y_pred_timestamp)

        # Save all relevant metrics for later analysis
        results.append({
            "test_subject": test_id,
            "true_jumps": true_jump_count,
            "predicted_jumps": predicted_jump_count,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "avg_coverage": avg_coverage,
            "avg_excess": avg_excess,
            "classification_report": report,
        })
    summary_df = pd.DataFrame(results)
    print("\n📈 Cross-validation summary:")
    print(summary_df[["test_subject", "true_jumps", "predicted_jumps", "true_positives", "avg_coverage", "avg_excess"]])
    summary_df.to_csv("loso_results_summary.csv", index=False)

if __name__ == "__main__":
    main()
