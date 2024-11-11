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
        data = np.append(data, [[time, float(row[17]), float(row[2]), float(row[5])]], axis=0) # use row[17] for altr
        # Stop at apogee
        if time > 7:
           break
data = data[1:]

# Process noise for accel: 10.05505052915663, calculated by np.mean(data[:, 3] + 9.81)
#aQ = 10.05505052915663
print(np.std(data[:, 3]))
aQ = 0.01

# Kalman filter initialization
def calcA(dt):
    return np.array([
        [1, dt, 0.5*dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])
C = np.array([
    [1, 0, 0],
    [0, 0, 1]
])

dt = 0.01
Q = np.array([
    [0.5*dt**2*aQ, 0, 0],
    [0, dt*aQ, 0],
    [0, 0, aQ]
])
R = np.array([
    [0.1, 0],
    [0, 0.01]
])
P = np.array([
    [0.01, 0, 0],
    [0, 0.01, 0],
    [0, 0, 0.01]
]) # Initial covariance matrix
x = np.array([[0], [0], [-9.81]]) # Initial state

# Update
out = np.array([[0, 0, 0, 0]])
for i in range(1, len(data)):
    # Prediction
    A = calcA(data[i][0] - data[i-1][0])
    x = A @ x
    P = A @ P @ A.T + Q

    # Update
    K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    x = x + K @ (np.array([[data[i][1]], [data[i][3]]]) - C @ x)
    P = (np.eye(3) - K @ C) @ P @ (np.eye(3) - K @ C).T + K @ R @ K.T

    out = np.append(out, [[data[i][0], x[0][0], x[1][0], x[2][0]]], axis=0)


# Plot
fig, axs = plt.subplots(3)
fig.suptitle('Flight data')

axs[0].plot(data[:,0], data[:,1])
axs[0].plot(out[:,0], out[:,1])
axs[0].set(ylabel='Position (m)', xlabel='Time (s)')
axs[0].grid()

axs[1].plot(data[:,0], data[:,2])
axs[1].plot(out[:,0], out[:,2])
axs[1].set(ylabel='Velocity (m/s)', xlabel='Time (s)')
axs[1].grid()

axs[2].plot(data[:,0], data[:,3])
axs[2].plot(out[:,0], out[:,3])
axs[2].set(ylabel='Acceleration (m/s²)', xlabel='Time (s)')
axs[2].grid()

plt.show()
