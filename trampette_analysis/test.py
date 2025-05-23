import pandas as pd
from data.loader import load_imu_data_and_annotations, combine_imu_data_and_annotations, get_jump
from data.filteringAndNoise import filter_and_noise_reduce
from visualizations.graphData import interactive_plot
from visualizations.orientation import visualize_puck_and_imu_df

def main():
    id = 7
    jump_nr = 12
    # Get all data
    sensor_df, annotations_df, jump_df = load_imu_data_and_annotations(id)
    sensor_df = combine_imu_data_and_annotations(sensor_df, annotations_df)
    # Get only the jump data
    jump_imu_df = get_jump(sensor_df, jump_df, jump_nr)
    jump_imu_df, q = filter_and_noise_reduce(jump_imu_df)
    print(jump_imu_df.head())
    print(jump_imu_df["label"].value_counts().sort_index())
    interactive_plot(jump_imu_df, columns=['lower_back_accx', 'lower_back_accy', 'lower_back_accz'], window_size=1000)
    visualize_puck_and_imu_df(q, jump_imu_df, columns=['lower_back_accx', 'lower_back_accy', 'lower_back_accz'], timestamp_col='timestamp')


if __name__ == "__main__":
    main()