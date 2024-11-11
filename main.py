import numpy as np
import matplotlib.pyplot as plt
import csv

# Read flight data
data = np.array([[0, 0, 0, 0]]) # t, x, v, a
with open('flight.csv', 'r') as f:
    reader = csv.reader(f)
    reader.__next__()
    for row in reader:
        time = float(row[0])/1000
        data = np.append(data, [[time, float(row[1]), float(row[2]), float(row[5])]], axis=0)
        # Stop at apogee
        if time > 7:
            break
data = data[1:]

# Plot
fig, axs = plt.subplots(3)
fig.suptitle('Flight data')

axs[0].plot(data[:,0], data[:,1])
axs[0].set(ylabel='Position (m)', xlabel='Time (s)')
axs[0].grid()

axs[1].plot(data[:,0], data[:,2])
axs[1].set(ylabel='Velocity (m/s)', xlabel='Time (s)')
axs[1].grid()

axs[2].plot(data[:,0], data[:,3])
axs[2].set(ylabel='Acceleration (m/s²)', xlabel='Time (s)')
axs[2].grid()

plt.show()
