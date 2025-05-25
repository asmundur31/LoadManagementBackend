from data.loader import load_imu_data_and_annotations, combine_imu_data_and_annotations, get_jump, load_jump_annotaitons
from data.filteringAndNoise import butterworth_filter_high, butterworth_filter_low, estimate_acc_bias, estimate_gyro_bias, correct_sensor_bias, fit_magnetometer_ellipsoid, get_orientation, rotate_data_into_earth_frame, remove_gravity, perform_fft
from visualizations.graphData import interactive_plot
from visualizations.orientation import visualize_puck_and_imu_df
from features.biomechanical import velocity, jerk, impulse, detect_takeoff_and_landing, jump_height, energy
from data.loader import get_subject_info
import pandas as pd
import numpy as np

def create_sensor_columns(placements, sensors=None):
    # Default to all sensors if none are specified
    if sensors is None:
        sensors = ['acc', 'gyr', 'mag']
    
    # Initialize lists to store the columns
    acc_cols, gyr_cols, mag_cols = [], [], []
    
    # Loop through each placement and generate columns
    for placement in placements:
        if 'acc' in sensors:
            acc_cols.extend([f'{placement}_accx', f'{placement}_accy', f'{placement}_accz'])
        if 'gyr' in sensors:
            gyr_cols.extend([f'{placement}_gyrox', f'{placement}_gyroy', f'{placement}_gyroz'])
        if 'mag' in sensors:
            mag_cols.extend([f'{placement}_magnx', f'{placement}_magny', f'{placement}_magnz'])
    
    return acc_cols, gyr_cols, mag_cols

def main():
    jump_annotations = load_jump_annotaitons()
    results = []
    for subject_id in jump_annotations["Subject ID"].unique():
        subject_jumps = jump_annotations[jump_annotations['Subject ID'] == subject_id]
        for jump_nr in subject_jumps["Jump Nr"]:
            subject_jump_features(subject_id, jump_nr, results)
    
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("data_public/processed/time_in_trampette_results.csv", index=False)


def subject_jump_features(id, jump_nr, results):
    # Get all data
    sensor_df, annotations_df, jump_df = load_imu_data_and_annotations(id)
    sensor_df = combine_imu_data_and_annotations(sensor_df, annotations_df)
    # Adjust the timestamp to start from 0
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    sensor_df['timestamp'] = (sensor_df['timestamp'] - sensor_df['timestamp'].iloc[0]).dt.total_seconds()
    # Get only the jump data
    raw_jump_imu_df, jump = get_jump(sensor_df, jump_df, jump_nr)
    jump_imu_df = raw_jump_imu_df.copy()

    video_jump_height = jump["ToF Height (m)"].iloc[0]
    video_ToF = jump["Video ToF (s)"].iloc[0]
    video_velocity = jump["Adjusted FSS (m/s)"].iloc[0]
    position = jump["Position"].iloc[0]

    # Remove gyro and mag bias
    placements = ['lower_back', 'shin', 'chest', 'thigh']
    for placement in placements:
        acc_cols, gyr_cols, mag_cols = create_sensor_columns([placement], sensors=['acc', 'gyr', 'mag'])
        gyro_bias = estimate_gyro_bias(jump_imu_df, gyr_cols, phases=[1]) # Static
        jump_imu_df = correct_sensor_bias(jump_imu_df, gyro_bias, gyr_cols)
        magn_bias = fit_magnetometer_ellipsoid(jump_imu_df[mag_cols].values) # All data
        jump_imu_df = correct_sensor_bias(jump_imu_df, magn_bias, mag_cols)
        #interactive_plot(jump_imu_df, columns=acc_cols, window_size=1000, title="Gyro and Mag bias removed", x_label="Time (s)", y_label="Acceleration (m/s^2)")
    
        # Low pass filter
        filtered_jump_imu_df = butterworth_filter_low(jump_imu_df, columns=acc_cols+gyr_cols+mag_cols, cutoff=10)
        #interactive_plot(jump_imu_df, columns=acc_cols, window_size=1000, title="Low pass filtered", x_label="Time (s)", y_label="Acceleration (m/s^2)")
    
        # Get orientation
        q = get_orientation(filtered_jump_imu_df, acc_cols, gyr_cols, mag_cols, fs=104.0)
        #visualize_puck_and_imu_df(q, jump_imu_df, columns=acc_cols, timestamp_col='timestamp')
    
        # Rotate into earth frame
        jump_imu_df_earth = rotate_data_into_earth_frame(jump_imu_df, q, acc_cols, gyr_cols, mag_cols)
        #interactive_plot(jump_imu_df_earth, columns=acc_cols, window_size=1000, title="Earth frame", x_label="Time (s)", y_label="Acceleration (m/s^2)")
    
        # Remove acc bias
        acc_bias = estimate_acc_bias(jump_imu_df_earth, acc_cols, phases=[1]) # Static
        jump_imu_df_earth = correct_sensor_bias(jump_imu_df_earth, acc_bias, acc_cols)
        #interactive_plot(jump_imu_df_earth, columns=acc_cols, window_size=1000, title="Acc bias removed", x_label="Time (s)", y_label="Acceleration (m/s^2)")
    
        # Remove gravity
        jump_imu_df_earth = remove_gravity(jump_imu_df_earth, acc_cols[2])
        #interactive_plot(jump_imu_df_earth, columns=acc_cols, window_size=1000, title="Gravity removed", x_label="Time (s)", y_label="Acceleration (m/s^2)")
        jump_imu_df = jump_imu_df_earth.copy()
    

    ### VELOCITY ###
    if 0:
        acc_cols, _, _ = create_sensor_columns(['lower_back', 'shin'], ['acc'])
        # Filter for horizontal acc
        jump_imu_df_earth = butterworth_filter_low(jump_imu_df_earth, columns=acc_cols, cutoff=20)
        jump_imu_df_earth = butterworth_filter_high(jump_imu_df_earth, columns=acc_cols, cutoff=0.1)
        # Calculate horizontal velocity for lower back
        jump_imu_df_earth = velocity(jump_imu_df_earth, raw_jump_imu_df, 'lower_back')
        # Calculate horizontal velocity for shin
        jump_imu_df_earth = velocity(jump_imu_df_earth, raw_jump_imu_df, 'shin', use_zupt=True)
        # Calculate the horizontal component
        jump_imu_df_earth['lower_back_velocity_h'] = np.sqrt(jump_imu_df_earth['lower_back_velocity_x']**2+jump_imu_df_earth['lower_back_velocity_y']**2)
        jump_imu_df_earth['shin_velocity_h'] = np.sqrt(jump_imu_df_earth['shin_velocity_x']**2 + jump_imu_df_earth['shin_velocity_y']**2)
        # Combine the velocities from the shin and lower back
        jump_imu_df_earth['average_velocity_h'] = jump_imu_df_earth['lower_back_velocity_h']*0.88+jump_imu_df_earth['shin_velocity_h']*0.12
        # Smooth the velocity
        jump_imu_df_earth['average_velocity_h'] = np.convolve(jump_imu_df_earth['average_velocity_h'], np.ones(20)/20, mode='same')
        # Get the velocity from the last part of the runup
        injump_mask = jump_imu_df_earth["label"].isin([3]) 
        fss = jump_imu_df_earth.loc[injump_mask, 'average_velocity_h'].iloc[-20:].mean()
        print(f"FSS: {fss}")
        print(f"Error: {np.abs(fss - video_velocity)}")
        results.append({
            "test_subject": id,
            "jump_nr": jump_nr,
            "landing": jump["Landing"].iloc[0],
            "trial": jump["Trial"].iloc[0],
            "fss": fss,
            "video_fss": video_velocity
        })
        

    ### JERK ###
    if 0:
        placements = ['lower_back']
        acc_cols, _, _ = create_sensor_columns(placements, ['acc'])
        jump_imu_df_earth = butterworth_filter_low(jump_imu_df_earth, columns=acc_cols, cutoff=10)
        jump_imu_df_earth = jerk(jump_imu_df_earth, 'lower_back', fs=104.0)
        #jump_imu_df_earth = butterworth_filter_low(jump_imu_df_earth, columns=[f"{placement}_jerk_magnitude"], cutoff=5)
        jump_imu_df_earth = butterworth_filter_high(jump_imu_df_earth, columns=[f"lower_back_jerk_magnitude"], cutoff=0.5)
        # Get the max jerk during takeoff
        takeoff_mask = jump_imu_df_earth["label"].isin([4, 5])
        max_takeoff_jerk = jump_imu_df_earth.loc[takeoff_mask, f"lower_back_jerk_magnitude"].max()
        print("Max jerk during takeoff: ", max_takeoff_jerk)
        # Get the max jerk during landing
        landing_mask = jump_imu_df_earth["label"].isin([7])
        max_landing_jerk = jump_imu_df_earth.loc[landing_mask, f"lower_back_jerk_magnitude"].max()
        print("Max jerk during landing: ", max_landing_jerk)
        results.append({
            "test_subject": id,
            "jump_nr": jump_nr,
            "landing": jump["Landing"].iloc[0],
            "trial": jump["Trial"].iloc[0],
            "max_takeoff_jerk": max_takeoff_jerk,
            "max_landing_jerk": max_landing_jerk
        })
        interactive_plot(jump_imu_df_earth, columns=[f'lower_back_jerk_magnitude'], window_size=1000, title="Jerk")
        interactive_plot(jump_imu_df_earth, columns=[f'lower_back_accx', 'lower_back_accy', 'lower_back_accz'], window_size=1000, title="Jerk")

    ### IMPULSE ###
    if 1:
        placements = ['lower_back']
        acc_cols, _, _ = create_sensor_columns(placements, ['acc'])
        jump_imu_df_earth = butterworth_filter_low(jump_imu_df_earth, columns=acc_cols, cutoff=10)
        jump_imu_df_earth = butterworth_filter_high(jump_imu_df_earth, columns=acc_cols, cutoff=0.1)
        #interactive_plot(jump_imu_df_earth, columns=[f'lower_back_accz', 'label'], window_size=1000, title="Jerk")
        jump_imu_df_earth = impulse(jump_imu_df_earth, id, 'lower_back', fs=104.0)
        jump_imu_df_earth = butterworth_filter_high(jump_imu_df_earth, columns=['lower_back_vertical_impulse'], cutoff=0.5)
        # Get the net impulse during takeoff
        takeoff_mask = jump_imu_df_earth["label"].isin([4, 5])
        takeoff_start = jump_imu_df_earth.loc[takeoff_mask, "lower_back_vertical_impulse"].iloc[0]
        takeoff_end = jump_imu_df_earth.loc[takeoff_mask, "lower_back_vertical_impulse"].min()
        net_takeoff_impulse = takeoff_end - takeoff_start
        print("Net impulse during takeoff: ", net_takeoff_impulse)
        # Get the net impulse during landing
        landing_mask = jump_imu_df_earth["label"].isin([7])
        landing_start = jump_imu_df_earth.loc[landing_mask, "lower_back_vertical_impulse"].iloc[0]
        landing_end = jump_imu_df_earth.loc[landing_mask, "lower_back_vertical_impulse"].min()
        net_landing_impulse = landing_end - landing_start
        print("Ner impulse during landing: ", net_landing_impulse)
        results.append({
            "test_subject": id,
            "jump_nr": jump_nr,
            "landing": jump["Landing"].iloc[0],
            "trial": jump["Trial"].iloc[0],
            "net_takeoff_impulse": net_takeoff_impulse,
            "net_landing_impulse": net_landing_impulse
        })
        #interactive_plot(jump_imu_df_earth, columns=['lower_back_vertical_impulse', 'lower_back_accx', 'lower_back_accy', 'lower_back_accz'], window_size=1000, title="Vertical impulse")
    
    
    ### JUMP HEIGHT ###
    if 0:
        placements = ['lower_back', 'shin', 'chest', 'thigh']
        acc_cols, _, _ = create_sensor_columns(placements, ['acc'])
        jump_height_values = {
            'lower_back': 0.0,
            'shin': 0.0,
            'chest': 0.0,
            'thigh': 0.0
        }
        tof_values = {
            'lower_back': 0.0,
            'shin': 0.0,
            'chest': 0.0,
            'thigh': 0.0
        }
        for placement in placements:
            jump_imu_df_earth[f"{placement}_acc_magnitude"] = np.sqrt(jump_imu_df_earth[f'{placement}_accx']**2 + jump_imu_df_earth[f'{placement}_accy']**2 + jump_imu_df_earth[f'{placement}_accz']**2)
            jump_imu_df_earth = butterworth_filter_low(jump_imu_df_earth, columns=[f"{placement}_acc_magnitude"], cutoff=10)
            jump_imu_df_earth = butterworth_filter_high(jump_imu_df_earth, columns=[f"{placement}_acc_magnitude"], cutoff=0.5)
            takeoff_idx, landing_idx = detect_takeoff_and_landing(jump_imu_df_earth, placement)
            #interactive_plot(jump_imu_df_earth, columns=[f'{placement}_acc_magnitude'], window_size=1000, title="Vertical impulse")
            error = 5.0
            if takeoff_idx and landing_idx:
                tof_values[placement] = jump_imu_df_earth['timestamp'].iloc[landing_idx] - jump_imu_df_earth['timestamp'].iloc[takeoff_idx]
                print("ToF based on acc: ", tof_values[placement])
                jump_height_values[placement] = jump_height(tof_values[placement], id) 
                print("Jump height based on acc: ", jump_height_values[placement])
                error = np.abs(jump_height_values[placement]-video_jump_height)
            else:
                print("No jump detected")

            print("Error: ", error)

        results.append({
            "test_subject": id,
            "jump_nr": jump_nr,
            "landing": jump["Landing"].iloc[0],
            "trial": jump["Trial"].iloc[0],
            "lower_back_jump_height": jump_height_values['lower_back'],
            "lower_back_tof": tof_values['lower_back'],
            "shin_jump_height": jump_height_values['shin'],
            "shin_tof": tof_values['shin'],
            "chest_jump_height": jump_height_values['chest'],
            "chest_tof": tof_values['chest'],
            "thigh_jump_height": jump_height_values['thigh'],
            "thigh_tof": tof_values['thigh'],
            "video_jump_height": video_jump_height,
            "video_tof": video_ToF
        })

    ### ENERGY ###
    if 0:
        placements = ['lower_back']
        _, gyr_cols, _ = create_sensor_columns(placements, ['gyro'])
        jump_imu_df_earth = butterworth_filter_low(jump_imu_df_earth, columns=gyr_cols, cutoff=10)

        subject_info = get_subject_info(id)
        s_mass = subject_info['Weight (kg)'][id-7] 

        flight_mask = jump_imu_df_earth["label"].isin([6])
        flight_phase = jump_imu_df_earth.loc[flight_mask]
        kinetic_energy, potential_energy, rotational_energy = energy(video_velocity, video_jump_height, s_mass, position, flight_phase)
        print(f"KE = {kinetic_energy}")
        print(f"PE = {potential_energy}")
        print(f"RE = {rotational_energy}")
        
        results.append({
            "test_subject": id,
            "jump_nr": jump_nr,
            "landing": jump["Landing"].iloc[0],
            "trial": jump["Trial"].iloc[0],
            "position": position,
            "KE": kinetic_energy,
            "PE": potential_energy,
            "RE": rotational_energy
        })
 

if __name__ == "__main__":
    main()