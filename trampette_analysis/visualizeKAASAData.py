import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

user_id = 4
recording = 'Testing protocol 2024-12-17'
sensor = '233830000584'
file_path = f'./data/raw/{user_id}/{recording}/2024_12_17-19_10_58/{sensor}/Acc-2024_12_17-19_10_58.csv'
with open(file_path, 'r') as file:
    header_info = file.readline().strip()

data = pd.read_csv(file_path, skiprows=1)

x_vals = data['AccX']
y_vals = data['AccY']
z_vals = data['AccZ']

# Find the max and min values of the data
# X-axis
max_x = x_vals.max()
min_x = x_vals.min()
max_x_index = x_vals.idxmax()
min_x_index = x_vals.idxmin()

# Y-axis
max_y = y_vals.max()
min_y = y_vals.min()
max_y_index = y_vals.idxmax()
min_y_index = y_vals.idxmin()

# Z-axis
max_z = z_vals.max()
min_z = z_vals.min()
max_z_index = z_vals.idxmax()
min_z_index = z_vals.idxmin()

# Display results
print(f"X-axis: max = {max_x} (index = {max_x_index}), min = {min_x} (index = {min_x_index})")
print(f"Y-axis: max = {max_y} (index = {max_y_index}), min = {min_y} (index = {min_y_index})")
print(f"Z-axis: max = {max_z} (index = {max_z_index}), min = {min_z} (index = {min_z_index})")

# Count the number of samples around max value of sensor
threshold = 156.5

# Count values above 156 or below -156 for each axis
count_max_x = (x_vals > threshold).sum()
count_max_y = (y_vals > threshold).sum()
count_max_z = (z_vals > threshold).sum()
count_min_x = (x_vals < -threshold).sum()
count_min_y = (y_vals < -threshold).sum()
count_min_z = (z_vals < -threshold).sum()

# Display results
print(f"Number of values above {threshold} or below {-threshold}:")
print(f"max X-axis: {count_max_x}")
print(f"max Y-axis: {count_max_y}")
print(f"max Z-axis: {count_max_z}")
print(f"min X-axis: {count_min_x}")
print(f"min Y-axis: {count_min_y}")
print(f"min Z-axis: {count_min_z}")

threshold = 147

def find_longest_consecutive_above_threshold(series, threshold):
    # Create a boolean mask for values above the threshold
    mask = series > threshold
    
    # Use pandas to identify consecutive groups using `cumsum`
    groups = (mask != mask.shift()).cumsum()
    
    # Filter only groups where the condition is True
    group_sizes = mask.groupby(groups).sum()  # Count True values in each group
    max_group = group_sizes.idxmax()          # Group with the max size
    max_size = group_sizes.max()              # Size of the max group
    
    # Find the starting index of this group
    if max_size > 0:
        start_index = series.index[(groups == max_group) & mask][0]
    else:
        start_index = None  # No values above the threshold
    
    return max_size, start_index

def find_longest_consecutive_below_threshold(series, threshold):
    # Create a boolean mask for values outside the thresholds
    mask = series < -threshold
    
    # Identify consecutive groups
    groups = (mask != mask.shift()).cumsum()
    
    # Filter only groups where the condition is True
    group_sizes = mask.groupby(groups).sum()  # Count True values in each group
    max_group = group_sizes.idxmax()          # Group with the max size
    max_size = group_sizes.max()              # Size of the max group
    
    # Find the starting index of this group
    if max_size > 0:
        start_index = series.index[(groups == max_group) & mask][0]
    else:
        start_index = None  # No values outside the thresholds
    
    return max_size, start_index

# Apply the function to each axis
max_consecutive_x, start_index_x_below = find_longest_consecutive_below_threshold(x_vals, threshold)
max_consecutive_y, start_index_y_below = find_longest_consecutive_below_threshold(y_vals, threshold)
max_consecutive_z, start_index_z_below = find_longest_consecutive_below_threshold(z_vals, threshold)

# Display results
print("Longest in row for below threshold")
print(f"X-axis: max consecutive = {max_consecutive_x}, starting at index = {start_index_x_below}")
print(f"Y-axis: max consecutive = {max_consecutive_y}, starting at index = {start_index_y_below}")
print(f"Z-axis: max consecutive = {max_consecutive_z}, starting at index = {start_index_z_below}")

# Apply the function to each axis
max_consecutive_x, start_index_x_above = find_longest_consecutive_above_threshold(x_vals, threshold)
max_consecutive_y, start_index_y_above = find_longest_consecutive_above_threshold(y_vals, threshold)
max_consecutive_z, start_index_z_above = find_longest_consecutive_above_threshold(z_vals, threshold)

# Display results
print("Longest in row for above threshold")
print(f"X-axis: max consecutive = {max_consecutive_x}, starting at index = {start_index_x_above}")
print(f"Y-axis: max consecutive = {max_consecutive_y}, starting at index = {start_index_y_above}")
print(f"Z-axis: max consecutive = {max_consecutive_z}, starting at index = {start_index_z_above}")


# Function to plot histogram with dynamic number of buckets
def plot_histogram(series, num_buckets):
    # Create histogram bins
    bins = np.linspace(series.min(), series.max(), num_buckets + 1)
    
    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of Values with {num_buckets} Buckets")
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Example usage: plot histograms for x_vals, y_vals, and z_vals
num_buckets = 32  # Change this dynamically
plot_histogram(x_vals, num_buckets)
plot_histogram(y_vals, num_buckets)
plot_histogram(z_vals, num_buckets)

# Then we set the timeto look at
start = min_x_index - 2*104
seconds = 4
# Assuming 104 Hz
end = start + seconds*104
plt.figure(figsize=(10, 6))
# Plot x, y, and z over index
plt.plot(x_vals[start:end], label='x', marker='o')
plt.plot(y_vals[start:end], label='y', marker='o')
plt.plot(z_vals[start:end], label='z', marker='o')
# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Acc x, y, z from Shin')
plt.legend()
# Show the plot
plt.show()