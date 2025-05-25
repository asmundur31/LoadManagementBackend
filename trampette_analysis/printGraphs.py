from visualizations.graphData import plot_jump_intensity_results, plot_landing_type_results, plot_jump_intensity_energy_results, plot_bland_altman
from features.statistical import calculate_icc, perform_t_test, perform_anova, perform_tukey_test, pearsons_correlation
import pingouin as pg
import numpy as np
import pandas as pd
from itertools import zip_longest

def main():
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    base_path = 'data_public/processed/'

    # Trampette time
    df = pd.read_csv(base_path+'time_in_trampette_results.csv')
    grouped = df.groupby("test_subject")["time_in_trampette"].apply(list)
    padded = list(zip_longest(*grouped.values, fillvalue=np.nan))
    time_in_trampette = np.array(padded).T
    submax_indices = [0, 1, 4, 5, 8, 9]
    nearmax_indices = [2, 3, 6, 7, 10, 11]

    sub_max_time = time_in_trampette[:, submax_indices]
    near_max_time = time_in_trampette[:, nearmax_indices]
    t_test = perform_t_test(sub_max_time, near_max_time)

    print("=================================")
    print("Time in trampette")
    print("T-test Results:")
    print(t_test)
    print()
    plot_jump_intensity_results(subjects, sub_max_time, near_max_time, title="Time in trampette results", y_label="Time (s)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    
    # Velocity intensity
    df = pd.read_csv(base_path+'velocity_results.csv')
    grouped = df.groupby("test_subject")["fss"].apply(list)
    padded = list(zip_longest(*grouped.values, fillvalue=np.nan))
    imu_velocity = np.array(padded).T

    grouped = df.groupby("test_subject")["video_fss"].apply(list)
    padded = list(zip_longest(*grouped.values, fillvalue=np.nan))
    video_velocity = np.array(padded).T

    submax_indices = [0, 1, 4, 5, 8, 9]
    nearmax_indices = [2, 3, 6, 7, 10, 11]

    sub_max_fss = imu_velocity[:, submax_indices]
    near_max_fss = imu_velocity[:, nearmax_indices]
    sub_max_fss_video = video_velocity[:, submax_indices]
    near_max_fss_video = video_velocity[:, nearmax_indices]
    
    # Reliability
    icc_sub =  calculate_icc(sub_max_fss, "Sub-maximal Velocity", model='ICC3')
    icc_near = calculate_icc(near_max_fss, "Near-maximal Velocity", model='ICC3')
    # Sensitivity
    t_test = perform_t_test(sub_max_fss, near_max_fss)
    t_test_video = perform_t_test(sub_max_fss_video, near_max_fss_video)

    #Validity
    t_test_validity = perform_t_test(video_velocity, imu_velocity)
    pearson_corr = pearsons_correlation(video_velocity, imu_velocity)
    RMSE = np.sqrt(np.nanmean((video_velocity - imu_velocity) ** 2))
    #plot_bland_altman(imu_velocity, video_velocity, title="FSS: IMU vs Video", y_label="difference between two measures (m/s)", x_label="average of two measures (m/s)")

    print("=================================")
    print("Velocity intensity")
    print("ICC Results:")
    print(icc_sub)
    print(icc_near)
    print("T-test Results:")
    print(t_test)
    print(t_test_video)
    print("T-test Validity Results:")
    print(t_test_validity)
    print("Pearson's Correlation Results:")
    print(pearson_corr)
    print("RMSE Results:")
    print(RMSE)
    print()
    #plot_jump_intensity_results(subjects, sub_max_fss, near_max_fss, title="IMU FSS results", y_label="Velocity (m/s)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    #plot_jump_intensity_results(subjects, sub_max_fss_video, near_max_fss_video, title="Video FSS results", y_label="Velocity (m/s)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    
    # Jerk intensity
    df = pd.read_csv(base_path+'jerk_results.csv')
    sub_filtered = df[df["trial"] == "Sub max"]
    takeoff_grouped = sub_filtered.groupby("test_subject")["max_takeoff_jerk"].apply(list)
    landing_grouped = sub_filtered.groupby("test_subject")["max_landing_jerk"].apply(list)
    takeoff_padded = list(zip_longest(*takeoff_grouped.values, fillvalue=np.nan))
    landing_padded = list(zip_longest(*landing_grouped.values, fillvalue=np.nan))
    sub_max_takeoff_jerk = np.array(takeoff_padded).T
    sub_max_landing_jerk = np.array(landing_padded).T

    near_filtered = df[df["trial"] == "Near max"]
    takeoff_grouped = near_filtered.groupby("test_subject")["max_takeoff_jerk"].apply(list)
    landing_grouped = near_filtered.groupby("test_subject")["max_landing_jerk"].apply(list)
    takeoff_padded = list(zip_longest(*takeoff_grouped.values, fillvalue=np.nan))
    landing_padded = list(zip_longest(*landing_grouped.values, fillvalue=np.nan))
    near_max_takeoff_jerk = np.array(takeoff_padded).T
    near_max_landing_jerk = np.array(landing_padded).T

    # Reliability
    icc_sub_takeoff =  calculate_icc(sub_max_takeoff_jerk, "Sub-maximal takeoff jerk", model='ICC3')
    icc_near_takeoff = calculate_icc(near_max_takeoff_jerk, "Near-maximal takeoff jerk", model='ICC3')
    icc_sub_landing =  calculate_icc(sub_max_landing_jerk, "Sub-maximal landing jerk", model='ICC3')
    icc_near_landing = calculate_icc(near_max_landing_jerk, "Near-maximal landing jerk", model='ICC3')
    # Sensitivity
    t_test_takeoff = perform_t_test(sub_max_takeoff_jerk, near_max_takeoff_jerk)
    t_test_landing = perform_t_test(sub_max_landing_jerk, near_max_landing_jerk)

    print("=================================")
    print("Jerk intensity")
    print("ICC Results:")
    print(icc_sub_takeoff)
    print(icc_near_takeoff)
    print(icc_sub_landing)
    print(icc_near_landing)
    print("T-test Results:")
    print(t_test_takeoff)
    print(t_test_landing)
    print()
    #plot_jump_intensity_results(subjects, sub_max_takeoff_jerk, near_max_takeoff_jerk, title="Peak jerk takeoff results", y_label="Jerk (m/s^3)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    #plot_jump_intensity_results(subjects, sub_max_landing_jerk, near_max_landing_jerk, title="Peak jerk landing results", y_label="Jerk (m/s^3)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])


    # Jerk landing type
    comp_filtered = df[df["landing"] == "Comp"]
    semi_filtered = df[df["landing"] == "Semi"]
    soft_filtered = df[df["landing"] == "Soft"]
    comp_takeoff_grouped = comp_filtered.groupby("test_subject")["max_takeoff_jerk"].apply(list)
    semi_takeoff_grouped = semi_filtered.groupby("test_subject")["max_takeoff_jerk"].apply(list)
    soft_takeoff_grouped = soft_filtered.groupby("test_subject")["max_takeoff_jerk"].apply(list)
    comp_takeoff_padded = list(zip_longest(*comp_takeoff_grouped.values, fillvalue=np.nan))
    semi_takeoff_padded = list(zip_longest(*semi_takeoff_grouped.values, fillvalue=np.nan))
    soft_takeoff_padded = list(zip_longest(*soft_takeoff_grouped.values, fillvalue=np.nan))
    comp_takeoff_jerk = np.array(comp_takeoff_padded).T
    semi_takeoff_jerk = np.array(semi_takeoff_padded).T
    soft_takeoff_jerk = np.array(soft_takeoff_padded).T

    comp_filtered = df[df["landing"] == "Comp"]
    semi_filtered = df[df["landing"] == "Semi"]
    soft_filtered = df[df["landing"] == "Soft"]
    comp_landing_grouped = comp_filtered.groupby("test_subject")["max_landing_jerk"].apply(list)
    semi_landing_grouped = semi_filtered.groupby("test_subject")["max_landing_jerk"].apply(list)
    soft_landing_grouped = soft_filtered.groupby("test_subject")["max_landing_jerk"].apply(list)
    comp_landing_padded = list(zip_longest(*comp_landing_grouped.values, fillvalue=np.nan))
    semi_landing_padded = list(zip_longest(*semi_landing_grouped.values, fillvalue=np.nan))
    soft_landing_padded = list(zip_longest(*soft_landing_grouped.values, fillvalue=np.nan))
    comp_landing_jerk = np.array(comp_landing_padded).T
    semi_landing_jerk = np.array(semi_landing_padded).T
    soft_landing_jerk = np.array(soft_landing_padded).T 
    

    # Reliability
    icc_comp_takeoff = calculate_icc(comp_takeoff_jerk, "Comp Jerk", model='ICC3')
    icc_semi_takeoff = calculate_icc(semi_takeoff_jerk, "Semi Jerk", model='ICC3')
    icc_soft_takeoff = calculate_icc(soft_takeoff_jerk, "Soft Jerk", model='ICC3')
    icc_comp_landing = calculate_icc(comp_landing_jerk, "Comp Jerk", model='ICC3')
    icc_semi_landing = calculate_icc(semi_landing_jerk, "Semi Jerk", model='ICC3')
    icc_soft_landing = calculate_icc(soft_landing_jerk, "Soft Jerk", model='ICC3')
    # Sensitivity
    anova_result_takeoff = perform_anova(comp_takeoff_jerk, semi_takeoff_jerk, soft_takeoff_jerk)
    anova_result_landing = perform_anova(comp_landing_jerk, semi_landing_jerk, soft_landing_jerk)
    # If there is statistical significance, perform Tukey's HSD test
    tuckey_result_takeoff = perform_tukey_test(comp_takeoff_jerk, semi_takeoff_jerk, soft_takeoff_jerk, feature_name="Jerk")
    tuckey_result_landing = perform_tukey_test(comp_landing_jerk, semi_landing_jerk, soft_landing_jerk, feature_name="Jerk")
    print("=================================")
    print("Jerk landing type")
    print("ICC Results:")
    print(icc_comp_takeoff)
    print(icc_semi_takeoff)
    print(icc_soft_takeoff)
    print(icc_comp_landing)
    print(icc_semi_landing)
    print(icc_soft_landing)
    print("ANOVA Results:")
    print(anova_result_takeoff)
    print(anova_result_landing)
    print("ONLY IF ANOVA IS SIGNIFICANT:")
    print("Tukey Results:")
    print(tuckey_result_takeoff)
    print(tuckey_result_landing)
    print()
    #plot_landing_type_results(subjects, comp_takeoff_jerk, semi_takeoff_jerk, soft_takeoff_jerk, title="Peak jerk takeoff results", y_label="Jerk (m/s^3)", x_label="Subjects", legend_labels=["Competition", "Semi", "Soft"])
    #plot_landing_type_results(subjects, comp_landing_jerk, semi_landing_jerk, soft_landing_jerk, title="Peak jerk landing results", y_label="Jerk (m/s^3)", x_label="Subjects", legend_labels=["Competition", "Semi", "Soft"])
    
    # Impulse intensity
    df = pd.read_csv(base_path+'impulse_results.csv')
    sub_filtered = df[df["trial"] == "Sub max"]
    takeoff_grouped = sub_filtered.groupby("test_subject")["net_takeoff_impulse"].apply(list)
    landing_grouped = sub_filtered.groupby("test_subject")["net_landing_impulse"].apply(list)
    takeoff_padded = list(zip_longest(*takeoff_grouped.values, fillvalue=np.nan))
    landing_padded = list(zip_longest(*landing_grouped.values, fillvalue=np.nan))
    sub_max_takeoff_impulse = -np.array(takeoff_padded).T
    sub_max_landing_impulse = -np.array(landing_padded).T

    near_filtered = df[df["trial"] == "Near max"]
    takeoff_grouped = near_filtered.groupby("test_subject")["net_takeoff_impulse"].apply(list)
    landing_grouped = near_filtered.groupby("test_subject")["net_landing_impulse"].apply(list)
    takeoff_padded = list(zip_longest(*takeoff_grouped.values, fillvalue=np.nan))
    landing_padded = list(zip_longest(*landing_grouped.values, fillvalue=np.nan))
    near_max_takeoff_impulse = -np.array(takeoff_padded).T
    near_max_landing_impulse = -np.array(landing_padded).T

    # Reliability
    icc_sub_takeoff = calculate_icc(sub_max_takeoff_impulse, "Sub-maximal takeoff impulse", model='ICC3')
    icc_near_takeoff = calculate_icc(near_max_takeoff_impulse, "Near-maximal takeoff impulse", model='ICC3')
    icc_sub_landing = calculate_icc(sub_max_landing_impulse, "Sub-maximal landing impulse", model='ICC3')
    icc_near_landing = calculate_icc(near_max_landing_impulse, "Near-maximal landing impulse", model='ICC3')
    # Sensitivity
    t_test_takeoff = perform_t_test(sub_max_takeoff_impulse, near_max_takeoff_impulse)
    t_test_landing = perform_t_test(sub_max_landing_impulse, near_max_landing_impulse)

    print("=================================")
    print("impulse intensity")
    print("ICC Results:")
    print(icc_sub_takeoff)
    print(icc_near_takeoff)
    print(icc_sub_landing)
    print(icc_near_landing)
    print("T-test Results:")
    print(t_test_takeoff)
    print(t_test_landing)
    print()
    plot_jump_intensity_results(subjects, sub_max_takeoff_impulse, near_max_takeoff_impulse, title="Net impulse takeoff results", y_label="Impulse (Ns)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    plot_jump_intensity_results(subjects, sub_max_landing_impulse, near_max_landing_impulse, title="Net impulse landing results", y_label="Impulse (Ns)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])


    # impulse landing type
    comp_filtered = df[df["landing"] == "Comp"]
    semi_filtered = df[df["landing"] == "Semi"]
    soft_filtered = df[df["landing"] == "Soft"]
    comp_takeoff_grouped = comp_filtered.groupby("test_subject")["net_takeoff_impulse"].apply(list)
    semi_takeoff_grouped = semi_filtered.groupby("test_subject")["net_takeoff_impulse"].apply(list)
    soft_takeoff_grouped = soft_filtered.groupby("test_subject")["net_takeoff_impulse"].apply(list)
    comp_takeoff_padded = list(zip_longest(*comp_takeoff_grouped.values, fillvalue=np.nan))
    semi_takeoff_padded = list(zip_longest(*semi_takeoff_grouped.values, fillvalue=np.nan))
    soft_takeoff_padded = list(zip_longest(*soft_takeoff_grouped.values, fillvalue=np.nan))
    comp_takeoff_impulse = -np.array(comp_takeoff_padded).T
    semi_takeoff_impulse = -np.array(semi_takeoff_padded).T
    soft_takeoff_impulse = -np.array(soft_takeoff_padded).T

    comp_filtered = df[df["landing"] == "Comp"]
    semi_filtered = df[df["landing"] == "Semi"]
    soft_filtered = df[df["landing"] == "Soft"]
    comp_landing_grouped = comp_filtered.groupby("test_subject")["net_landing_impulse"].apply(list)
    semi_landing_grouped = semi_filtered.groupby("test_subject")["net_landing_impulse"].apply(list)
    soft_landing_grouped = soft_filtered.groupby("test_subject")["net_landing_impulse"].apply(list)
    comp_landing_padded = list(zip_longest(*comp_landing_grouped.values, fillvalue=np.nan))
    semi_landing_padded = list(zip_longest(*semi_landing_grouped.values, fillvalue=np.nan))
    soft_landing_padded = list(zip_longest(*soft_landing_grouped.values, fillvalue=np.nan))
    comp_landing_impulse = -np.array(comp_landing_padded).T
    semi_landing_impulse = -np.array(semi_landing_padded).T
    soft_landing_impulse = -np.array(soft_landing_padded).T 
    

    # Reliability
    icc_comp_takeoff = calculate_icc(comp_takeoff_impulse, "Comp impulse", model='ICC3')
    icc_semi_takeoff = calculate_icc(semi_takeoff_impulse, "Semi impulse", model='ICC3')
    icc_soft_takeoff = calculate_icc(soft_takeoff_impulse, "Soft impulse", model='ICC3')
    icc_comp_landing = calculate_icc(comp_landing_impulse, "Comp impulse", model='ICC3')
    icc_semi_landing = calculate_icc(semi_landing_impulse, "Semi impulse", model='ICC3')
    icc_soft_landing = calculate_icc(soft_landing_impulse, "Soft impulse", model='ICC3')
    # Sensitivity
    anova_result_takeoff = perform_anova(comp_takeoff_impulse, semi_takeoff_impulse, soft_takeoff_impulse)
    anova_result_landing = perform_anova(comp_landing_impulse, semi_landing_impulse, soft_landing_impulse)
    # If there is statistical significance, perform Tukey's HSD test
    tuckey_result_takeoff = perform_tukey_test(comp_takeoff_impulse, semi_takeoff_impulse, soft_takeoff_impulse, feature_name="impulse")
    tuckey_result_landing = perform_tukey_test(comp_landing_impulse, semi_landing_impulse, soft_landing_impulse, feature_name="impulse")
    print("=================================")
    print("impulse landing type")
    print("ICC Results:")
    print(icc_comp_takeoff)
    print(icc_semi_takeoff)
    print(icc_soft_takeoff)
    print(icc_comp_landing)
    print(icc_semi_landing)
    print(icc_soft_landing)
    print("ANOVA Results:")
    print(anova_result_takeoff)
    print(anova_result_landing)
    print("ONLY IF ANOVA IS SIGNIFICANT:")
    print("Tukey Results:")
    print(tuckey_result_takeoff)
    print(tuckey_result_landing)
    print()
    plot_landing_type_results(subjects, comp_takeoff_impulse, semi_takeoff_impulse, soft_takeoff_impulse, title="Net impulse takeoff results", y_label="Impulse (Ns)", x_label="Subjects", legend_labels=["Competition", "Semi", "Soft"])
    plot_landing_type_results(subjects, comp_landing_impulse, semi_landing_impulse, soft_landing_impulse, title="Net impulse landing results", y_label="Impulse (Ns)", x_label="Subjects", legend_labels=["Competition", "Semi", "Soft"])
    
    # Jump height
    df = pd.read_csv(base_path+'jump_height_all_results.csv')
    grouped = df.groupby("test_subject")["lower_back_jump_height"].apply(list)
    padded = list(zip_longest(*grouped.values, fillvalue=np.nan))
    imu_jump_height = np.array(padded).T

    grouped = df.groupby("test_subject")["video_jump_height"].apply(list)
    padded = list(zip_longest(*grouped.values, fillvalue=np.nan))
    video_jump_height = np.array(padded).T

    submax_indices = [0, 1, 4, 5, 8, 9]
    nearmax_indices = [2, 3, 6, 7, 10, 11]

    sub_max_height = imu_jump_height[:, submax_indices]
    near_max_height = imu_jump_height[:, nearmax_indices]
    sub_max_height_video = video_jump_height[:, submax_indices]
    near_max_height_video = video_jump_height[:, nearmax_indices]

    # Reliability
    icc_sub =  calculate_icc(sub_max_height, "Sub-maximal Height", model='ICC3')
    icc_near = calculate_icc(near_max_height, "Near-maximal Height", model='ICC3')
    # Sensitivity
    t_test = perform_t_test(sub_max_height, near_max_height)
    t_test_video = perform_t_test(sub_max_height_video, near_max_height_video)

    # Validity
    t_test_validity = perform_t_test(video_jump_height, imu_jump_height)
    pearson_corr = pearsons_correlation(video_jump_height, imu_jump_height)
    RMSE = np.sqrt(np.nanmean((video_jump_height - imu_jump_height) ** 2))
    #plot_bland_altman(imu_jump_height, video_jump_height, title="Bland-Altman plot: IMU vs Video", y_label="difference between two measures (m)", x_label="average of two measures (m)")

    print("=================================")
    print("Jump height")
    print("ICC Results:")
    print(icc_sub)
    print(icc_near)
    print("T-test Results:")
    print(t_test)
    print(t_test_video)
    print("T-test Validity Results:")
    print(t_test_validity)
    print("Pearson's Correlation Results:")
    print(pearson_corr)
    print("RMSE Results:")
    print(RMSE)
    print()

    plot_jump_intensity_results(subjects, sub_max_height, near_max_height, title="IMU jump height results", y_label="Height (m)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    plot_jump_intensity_results(subjects, sub_max_height_video, near_max_height_video, title="Video jump height results", y_label="Height (m)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    
    # Energy intensity
    df = pd.read_csv(base_path+'energy_results.csv')
    print(df)
    def build_array_for_position(df, column, value, value_column='value', all_subjects=None):
        filtered = df[df[column] == value]
        if all_subjects is None:
            all_subjects = df['test_subject'].unique()
        grouped = (
            filtered.groupby('test_subject')[value_column]
            .apply(list)
            .reindex(all_subjects)
        )
        grouped = grouped.apply(lambda x: [np.nan] if isinstance(x, float) and np.isnan(x) else x)
        padded = list(zip_longest(*grouped.values, fillvalue=np.nan))
        return np.array(padded).T

    all_subjects = df['test_subject'].unique()

    tuck_re = build_array_for_position(df, 'position', 'Tuck', value_column='RE', all_subjects=all_subjects)
    pike_re = build_array_for_position(df, 'position', 'Pike', value_column='RE', all_subjects=all_subjects)
    straight_re = build_array_for_position(df, 'position', 'Straight', value_column='RE', all_subjects=all_subjects)

    comp_re = build_array_for_position(df, 'landing', 'Competition', value_column='RE')
    semi_re = build_array_for_position(df, 'landing', 'Semi in Soft pit', value_column='RE')
    soft_re = build_array_for_position(df, 'landing', 'Soft pit', value_column='RE')

    sub_max_ke = build_array_for_position(df, 'trial', 'Sub max', value_column='KE')
    sub_max_pe = build_array_for_position(df, 'trial', 'Sub max', value_column='PE')
    sub_max_re = build_array_for_position(df, 'trial', 'Sub max', value_column='RE')

    near_max_ke = build_array_for_position(df, 'trial', 'Near max', value_column='KE')
    near_max_pe = build_array_for_position(df, 'trial', 'Near max', value_column='PE')
    near_max_re = build_array_for_position(df, 'trial', 'Near max', value_column='RE')

    print(sub_max_ke)

    #sub_filtered = df[df["trial"] == "Sub max"]
    #ke_grouped = sub_filtered.groupby("test_subject")["KE"].apply(list)
    #pe_grouped = sub_filtered.groupby("test_subject")["PE"].apply(list)
    #re_grouped = sub_filtered.groupby("test_subject")["RE"].apply(list)
    #ke_padded = list(zip_longest(*ke_grouped.values, fillvalue=np.nan))
    #pe_padded = list(zip_longest(*pe_grouped.values, fillvalue=np.nan))
    #re_padded = list(zip_longest(*re_grouped.values, fillvalue=np.nan))
    #sub_max_ke = np.array(ke_padded).T
    #sub_max_pe = np.array(pe_padded).T
    #sub_max_re = np.array(re_padded).T

    #near_filtered = df[df["trial"] == "Near max"]
    #ke_grouped = near_filtered.groupby("test_subject")["KE"].apply(list)
    #pe_grouped = near_filtered.groupby("test_subject")["PE"].apply(list)
    #re_grouped = near_filtered.groupby("test_subject")["RE"].apply(list)
    #ke_padded = list(zip_longest(*ke_grouped.values, fillvalue=np.nan))
    #pe_padded = list(zip_longest(*pe_grouped.values, fillvalue=np.nan))
    #re_padded = list(zip_longest(*re_grouped.values, fillvalue=np.nan))
    #near_max_ke = np.array(ke_padded).T
    #near_max_pe = np.array(pe_padded).T
    #near_max_re = np.array(re_padded).T

    sub_max_keA = sub_max_ke
    sub_max_peA = np.zeros_like(sub_max_ke)
    sub_max_reA = np.zeros_like(sub_max_ke)

    sub_max_keB = np.zeros_like(sub_max_pe)
    sub_max_peB = sub_max_pe
    sub_max_reB = sub_max_re
    
    near_max_keA = near_max_ke
    near_max_peA = np.zeros_like(near_max_ke)
    near_max_reA = np.zeros_like(near_max_ke)

    near_max_keB = np.zeros_like(near_max_pe)
    near_max_peB = near_max_pe
    near_max_reB = near_max_re

    sub_maxA = sub_max_keA+sub_max_peA+sub_max_reA
    sub_maxB = sub_max_keB+sub_max_peB+sub_max_reB
    near_maxA = near_max_keA+near_max_peA+near_max_reA
    near_maxB = near_max_keB+near_max_peB+near_max_reB 

    # Reliability
    icc_subA = calculate_icc(sub_maxA, "Sub-maximal A", model='ICC3')
    icc_subB = calculate_icc(sub_maxB, "Sub-maximal B", model='ICC3')
    icc_nearA = calculate_icc(near_maxA, "Near-maximal A", model='ICC3')
    icc_nearB = calculate_icc(near_maxB, "Near-maximal B", model='ICC3')
    # Reliability rotational energy
    icc_sub_re = calculate_icc(sub_max_re, "Sub-maximal RE", model='ICC3')
    icc_near_re = calculate_icc(near_max_re, "Near-maximal RE", model='ICC3')
    icc_tuck_re = calculate_icc(tuck_re, "Tuck RE", model='ICC3')
    icc_pike_re = calculate_icc(pike_re, "Pike RE", model='ICC3')
    icc_straight_re = calculate_icc(straight_re, "Straight RE", model='ICC3')
    # Sensitivity
    t_test_intensity_re = perform_t_test(sub_max_re, near_max_re)
    t_testA = perform_t_test(sub_maxA, near_maxA)
    t_testB = perform_t_test(sub_maxB, near_maxB)
    #anova_result_position_re = perform_anova(tuck_re, pike_re, straight_re)
    # If there is statistical significance, perform Tukey's HSD test
    #tuckey_result_position = perform_tukey_test(tuck_re, pike_re, straight_re, feature_name="Rotational energy")
    
    print("=================================")
    print("Energy intensity")
    print("ICC Results Rotational energy:")
    print(icc_sub_re)
    print(icc_near_re)
    print(icc_tuck_re)
    print(icc_pike_re)
    print(icc_straight_re)
    print("ICC Results:")
    print(icc_subA)
    print(icc_subB)
    print(icc_nearA)
    print(icc_nearB)
    print("T-test Results:")
    print(t_test_intensity_re)
    print(t_testA)
    print(t_testB)
    #print("ANOVA Results:")
    #print(anova_result_position_re)
    #print("ONLY IF ANOVA IS SIGNIFICANT:")
    #print("Tukey Results:")
    #print(tuckey_result_position)
    print()

    plot_landing_type_results(subjects, tuck_re, pike_re, straight_re, title="Rotational energy position results", y_label="Energy (J)", x_label="Subjects", legend_labels=["Tuck", "Pike", "Straight"])
    plot_jump_intensity_results(subjects, sub_max_re, near_max_re, title="Rotational energy intensity results", y_label="Energy (J)", x_label="Subjects", legend_labels=["Sub-maximal", "Near-maximal"])
    plot_landing_type_results(subjects, comp_re, semi_re, soft_re, title="Rotational energy landing results", y_label="Energy (J)", x_label="Subjects", legend_labels=["Competition", "Semi", "Soft"])
    plot_jump_intensity_energy_results(subjects, sub_maxA, near_maxA, sub_maxB, near_maxB, title="Energy results", y_label="Energy (J)", x_label="Subjects", legend_labels=["Sub-maximal A", "Near-maximal A", "Sub-maximal B", "Near-maximal B"])
if __name__ == "__main__":
    main()
