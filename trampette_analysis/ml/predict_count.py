import pandas as pd
import numpy as np
import joblib
from trampette_analysis.ml.models.jump_count_model import predict, count, timestamp_predictions
from trampette_analysis.data.preprocessing import segment_test_data 
from trampette_analysis.data.filteringAndNoise import butterworth_filter_high, butterworth_filter_low, downsample

def predict_count(df):
    if len(df)==0:
        return 0
    # Get here if the sensor is from the lower back
    acc_cols = ['lower_back_accx', 'lower_back_accy', 'lower_back_accz', 'chest_accx', 'chest_accy', 'chest_accz', 'thigh_accx', 'thigh_accy', 'thigh_accz', 'shin_accx', 'shin_accy', 'shin_accz']
    gyro_cols = ['lower_back_gyrox', 'lower_back_gyroy', 'lower_back_gyroz', 'chest_gyrox', 'chest_gyroy', 'chest_gyroz', 'thigh_gyrox', 'thigh_gyroy', 'thigh_gyroz', 'shin_gyrox', 'shin_gyroy', 'shin_gyroz']
    magn_cols = ['lower_back_magnx', 'lower_back_magny', 'lower_back_magnz', 'chest_magnx', 'chest_magny', 'chest_magnz', 'thigh_magnx', 'thigh_magny', 'thigh_magnz', 'shin_magnx', 'shin_magny', 'shin_magnz']

    # Filter data
    df = butterworth_filter_low(df, columns=acc_cols+gyro_cols+magn_cols, cutoff=20)
    df = butterworth_filter_high(df, columns=acc_cols+gyro_cols+magn_cols, cutoff=0.5)
    
    # Downsample
    df = downsample(df, 2)

    # Normalize with the saved scalar
    scaler = joblib.load('/app/data/models/final_jump_count_scaler.pkl')
    df_scaled = pd.DataFrame(
        scaler.transform(df[acc_cols+gyro_cols+magn_cols]),
        columns=acc_cols + gyro_cols + magn_cols
    )

    # If there are other columns (e.g., timestamps), merge back
    if 'timestamp' in df.columns:
        df_scaled['timestamp'] = df['timestamp'].values

    # Segment data
    X_new = segment_test_data(df_scaled)

    # Get the model
    model = joblib.load('/app/data/models/final_jump_count_model.pkl')
    # Predict
    print("Predict the number of jumps")
    batch_size = 500
    y_pred_list = []

    for i in range(0, X_new.shape[0], batch_size):
        batch = X_new[i:i+batch_size]
        y_pred_batch = predict(model, batch)
        y_pred_list.append(y_pred_batch)

    y_pred = np.concatenate(y_pred_list)
    
    print("Get timestamp predictions")
    y_pred_timestamp = timestamp_predictions(y_pred, len(df_scaled))
    print("Get the jump count")
    predicted_jump_count = count(y_pred_timestamp)
    print(f"Jump count = {predicted_jump_count}")
    return predicted_jump_count 