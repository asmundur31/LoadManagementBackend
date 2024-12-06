import matplotlib.pyplot as plt
import pandas as pd

with open('./data/Test KAASA/Third try with 104Hz/2024_09_12-18_07_28/200830001383/Acc-2024_09_12-18_07_28.csv', 'r') as file:
    header_info = file.readline().strips()

data = pd.read_csv('./data/Test KAASA/Third try with 104Hz/2024_09_12-18_07_28/200830001383/Acc-2024_09_12-18_07_28.csv', skiprows=1)
print(data.columns)
x_vals = data['AccX']
y_vals = data['AccY']
z_vals = data['AccZ']

start = 0 
seconds = 8
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