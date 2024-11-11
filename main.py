import numpy as np
import matplotlib.pyplot as plt
import csv

from predict import getApogee

# Read flight data
data = np.array([[0, 0, 0, 0, 0]]) # t, x, v, a, predicted alt
burnend = 0
with open('characterizationflight.csv', 'r') as f:
    reader = csv.reader(f)
    reader.__next__()
    for row in reader:
        time = float(row[0])/1000
        data = np.append(data, [[time, float(row[17]), float(row[2]), float(row[5]), getApogee(time, float(row[17]), float(row[2]), float(row[5]))]], axis=0) # use row[17] for altr
        # Stop at apogee
        if time > 3.2 and burnend == 0:
            burnend = len(data)
        if time > 7:
           break
data = data[1:]

# Process noise for accel: 10.05505052915663, calculated by np.mean(data[:, 3] + 9.81)
#aQ = 10.05505052915663
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
    [1, 0],
    [0, 0.01]
])
P = np.array([
    [0.01, 0, 0],
    [0, 0.01, 0],
    [0, 0, 0.01]
]) # Initial covariance matrix
x = np.array([[0], [0], [-9.81]]) # Initial state

# Update
out = np.array([[0, 0, 0, 0, 0]])
for i in range(1, len(data)):
    # Prediction
    A = calcA(data[i][0] - data[i-1][0])
    x = A @ x
    P = A @ P @ A.T + Q

    # Update
    K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    x = x + K @ (np.array([[data[i][1]], [data[i][3]]]) - C @ x)
    P = (np.eye(3) - K @ C) @ P @ (np.eye(3) - K @ C).T + K @ R @ K.T

    out = np.append(out, [[data[i][0], x[0][0], x[1][0], x[2][0], getApogee(data[i][0], x[0][0], x[1][0], x[2][0])]], axis=0)
out = out[1:]


# Plot
fig, axs = plt.subplots(4)
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

axs[3].plot(data[burnend:,0], data[burnend:,4])
axs[3].plot(out[burnend:,0], out[burnend:,4])
axs[3].plot(out[burnend:,0], np.zeros(len(out[burnend:,0])) + np.max(data[:, 1]), 'r--')
axs[3].set(ylabel='Predicted altitude (m)', xlabel='Time (s)')
axs[3].grid()

def rmse(y_pred):
  return np.sqrt(np.mean((np.max(data[:, 1]) - y_pred)**2))

print(f"Regular pre mean error: {rmse(data[burnend:,4])}")
print(f"Kalman pre mean error: {rmse(out[burnend:,4])}")

plt.show()
