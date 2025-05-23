import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def interactive_plot(df: pd.DataFrame, columns, vertical_lines=None, window_size=1000, title="Interactive Plot", x_label="Time", y_label="Value"):
    """
    Launch an interactive plot of selected columns in a scrolling window.
    
    Arrow keys:
    - Left / Right: Move window by 80% of window size
    - Up / Down: Increase / Decrease window size
    """
    step_size = int(window_size * 0.8)
    start = 0
    end = start + window_size

    fig, ax = plt.subplots()
    lines = {col: ax.plot([], [], label=col)[0] for col in columns}
    ax.legend()

    y_min = df[columns].min().min()
    y_max = df[columns].max().max()
    ax.set_ylim(y_min, y_max)

    def update_plot():
        nonlocal start, end
        start = max(0, min(start, len(df) - 1))
        end = min(start + window_size, len(df))
        x = df.index[start:end]
        for col in columns:
            lines[col].set_data(x, df[col].iloc[start:end])
        ax.set_xlim(x.min(), x.max())
        ax.set_title(f"{title} | Window: {start} to {end} | Window size: {window_size}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if vertical_lines:
            ax.axvline(x=[vertical_lines[0]], color='r', linestyle='--', label='Takeoff')
            ax.axvline(x=[vertical_lines[1]], color='g', linestyle='--', label='Landing')
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal start, end, window_size, step_size
        if event.key == 'right':
            start += step_size
        elif event.key == 'left':
            start -= step_size
        elif event.key == 'up':
            window_size = int(window_size * 1.2)
            step_size = int(window_size * 0.8)
        elif event.key == 'down':
            window_size = max(100, int(window_size * 0.8))
            step_size = int(window_size * 0.8)
        update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    plt.show()

def plot_jump_intensity_results(subjects, sub_max, near_max, title, y_label, x_label, legend_labels):
    x = np.arange(len(subjects))
    width = 0.35

    # Calculate means and standard deviations for each subject (row-wise)
    sub_max_mean = np.nanmean(sub_max, axis=1)
    near_max_mean = np.nanmean(near_max, axis=1)
    sub_max_std = np.nanstd(sub_max, axis=1)
    near_max_std = np.nanstd(near_max, axis=1)

    # Calculate overall means for sub_max and near_max (mean of all subjects)
    sub_max_mean_all = np.mean(sub_max_mean)
    near_max_mean_all = np.mean(near_max_mean)
    sub_max_std_all = np.mean(sub_max_std)
    near_max_std_all = np.mean(near_max_std)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width/2, sub_max_mean, width, yerr=sub_max_std, label=legend_labels[0], capsize=5)
    bar2 = ax.bar(x + width/2, near_max_mean, width, yerr=near_max_std, label=legend_labels[1], capsize=5)

    # Add the overall mean columns (at the end of the plot)
    ax.bar(len(subjects) - width/2, sub_max_mean_all, width, yerr=sub_max_std_all, label=f'Mean {legend_labels[0]}', capsize=5)
    ax.bar(len(subjects) + width/2, near_max_mean_all, width, yerr=near_max_std_all, label=f'Mean {legend_labels[1]}', capsize=5)

    # Labels and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(np.concatenate([x, [len(subjects)]]))
    ax.set_xticklabels(np.concatenate([subjects, ['Mean']]))
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_landing_type_results(subjects, comp, semi, soft, title, y_label, x_label, legend_labels):
    x = np.arange(len(subjects))
    width = 0.25

    comp_mean = np.nanmean(comp, axis=1)
    semi_mean = np.nanmean(semi, axis=1)
    soft_mean = np.nanmean(soft, axis=1)
    comp_std = np.nanstd(comp, axis=1)
    semi_std = np.nanstd(semi, axis=1)
    soft_std = np.nanstd(soft, axis=1)

    # Calculate overall means for sub_max and near_max (mean of all subjects)
    comp_mean_all = np.nanmean(comp_mean)
    semi_mean_all = np.nanmean(semi_mean)
    soft_mean_all = np.nanmean(soft_mean)
    comp_std_all = np.nanmean(comp_std)
    semi_std_all = np.nanmean(semi_std)
    soft_std_all = np.nanmean(soft_std)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width, comp_mean, width, yerr=comp_std, label=legend_labels[0], capsize=5)
    bar2 = ax.bar(x, semi_mean, width, yerr=semi_std, label=legend_labels[1], capsize=5)
    bar3 = ax.bar(x + width, soft_mean, width, yerr=soft_std, label=legend_labels[2], capsize=5)
    
    ax.bar(len(subjects) - width, comp_mean_all, width, yerr=comp_std_all, label=f'Mean {legend_labels[0]}', capsize=5)
    ax.bar(len(subjects), semi_mean_all, width, yerr=semi_std_all, label=f'Mean {legend_labels[1]}', capsize=5)
    ax.bar(len(subjects) + width, soft_mean_all, width, yerr=soft_std_all, label=f'Mean {legend_labels[2]}', capsize=5)

    # Labels and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(np.concatenate([x, [len(subjects)]]))
    ax.set_xticklabels(np.concatenate([subjects, ['Mean']]))
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_jump_intensity_energy_results(subjects, sub_maxA, sub_maxB, near_maxA, near_maxB, title, y_label, x_label, legend_labels):
    x = np.arange(len(subjects))
    width = 0.2

    # Calculate means and standard deviations for each subject (row-wise)
    sub_maxA_mean = np.nanmean(sub_maxA, axis=1)
    near_maxA_mean = np.nanmean(near_maxA, axis=1)
    sub_maxA_std = np.nanstd(sub_maxA, axis=1)
    near_maxA_std = np.nanstd(near_maxA, axis=1)
    sub_maxB_mean = np.nanmean(sub_maxB, axis=1)
    near_maxB_mean = np.nanmean(near_maxB, axis=1)
    sub_maxB_std = np.nanstd(sub_maxB, axis=1)
    near_maxB_std = np.nanstd(near_maxB, axis=1)

    # Calculate overall means for sub_max and near_max (mean of all subjects)
    sub_maxA_mean_all = np.mean(sub_maxA_mean)
    near_maxA_mean_all = np.mean(near_maxA_mean)
    sub_maxA_std_all = np.mean(sub_maxA_std)
    near_maxA_std_all = np.mean(near_maxA_std)
    sub_maxB_mean_all = np.mean(sub_maxB_mean)
    near_maxB_mean_all = np.mean(near_maxB_mean)
    sub_maxB_std_all = np.mean(sub_maxB_std)
    near_maxB_std_all = np.mean(near_maxB_std)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width*3/2, sub_maxA_mean, width, yerr=sub_maxA_std, label=legend_labels[0], capsize=5)
    bar2 = ax.bar(x - width/2, sub_maxB_mean, width, yerr=sub_maxB_std, label=legend_labels[1], capsize=5)
    bar3 = ax.bar(x + width/2, near_maxA_mean, width, yerr=near_maxA_std, label=legend_labels[2], capsize=5)
    bar4 = ax.bar(x + width*3/2, near_maxB_mean, width, yerr=near_maxB_std, label=legend_labels[3], capsize=5)

    ax.bar(len(subjects) - width*3/2, sub_maxA_mean_all, width, yerr=sub_maxA_std_all, label=f'Mean {legend_labels[0]}', capsize=5)
    ax.bar(len(subjects) - width/2, sub_maxB_mean_all, width, yerr=sub_maxB_std_all, label=f'Mean {legend_labels[1]}', capsize=5)
    ax.bar(len(subjects) + width/2, near_maxA_mean_all, width, yerr=near_maxA_std_all, label=f'Mean {legend_labels[2]}', capsize=5)
    ax.bar(len(subjects) + width*3/2, near_maxB_mean_all, width, yerr=near_maxB_std_all, label=f'Mean {legend_labels[3]}', capsize=5)

    # Labels and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(np.concatenate([x, [len(subjects)]]))
    ax.set_xticklabels(np.concatenate([subjects, ['Mean']]))
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_bland_altman(imu_data, video_data, title, y_label, x_label):
    # Convert to arrays
    imu_data = np.asarray(imu_data)
    video_data = np.asarray(video_data)

    # Flatten and remove NaNs
    imu_flat = imu_data.flatten()
    video_flat = video_data.flatten()
    valid = ~np.isnan(imu_flat) & ~np.isnan(video_flat)
    imu_flat = imu_flat[valid]
    video_flat = video_flat[valid]

    # Bland-Altman calculations
    mean = (imu_flat + video_flat) / 2
    diff = imu_flat - video_flat
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    upper = md + 1.96 * sd
    lower = md - 1.96 * sd

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(mean, diff, alpha=0.7)
    plt.axhline(md, color='gray', linestyle='--', label=f'Mean diff = {md:.2f}')
    plt.axhline(upper, color='red', linestyle='--', label=f'+1.96 SD = {upper:.2f}')
    plt.axhline(lower, color='red', linestyle='--', label=f'-1.96 SD = {lower:.2f}')

    # Annotate lines
    x_pos = np.max(mean)
    plt.text(x_pos, md, f'mean', color='gray', fontsize=8, va='bottom', ha='right')
    plt.text(x_pos, upper, f'+1.96 SD', color='red', fontsize=8, va='bottom', ha='right')
    plt.text(x_pos, lower, f'-1.96 SD', color='red', fontsize=8, va='top', ha='right')

    # Labels and layout
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()