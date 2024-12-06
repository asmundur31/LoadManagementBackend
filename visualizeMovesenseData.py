import matplotlib.pyplot as plt
import json
# We start vizualising the Movesense data
with open('./data/Test Movesense/2024-09-12-20-35/20240912T183547Z_233830000584_acc_stream.json', 'r') as file:
    data = json.load(file)

x_vals = []
y_vals = []
z_vals = []
for d in data['data']:
    acc_d = d['acc']
    acc_datapoints = acc_d['ArrayAcc']
    x_vals.extend([item['x'] for item in acc_datapoints])
    y_vals.extend([item['y'] for item in acc_datapoints])
    z_vals.extend([item['z'] for item in acc_datapoints])

start = 14200
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