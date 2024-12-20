import numpy as np
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from predict import getApogee

STOP_APOGEE = True

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
        if time > 7 and STOP_APOGEE:
           break
        elif time > 37 and not STOP_APOGEE:
            break
data = data[1:]

# Process noise for accel: 0.25, calculated by np.mean(data[:, 3])
print(np.mean(data[:, 3]), np.std(data[:, 3]), np.mean(abs(data[:, 3])))
aQ = 0.25

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
    [0.5*dt**2*aQ**2 + dt*aQ**2, 0, 0],
    [0, dt*aQ**2, 0],
    [0, 0, aQ**2]
])
R = np.array([
    [1, 0],
    [0, 0.01]
])
P = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]) # Initial covariance matrix, its a diagonal matrix but all the diagonals are 0 because the initial state is known
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

# Convert data to dataframe
df = pd.DataFrame(data, columns=['Time (s)', 'Position (m)', 'Velocity (m/s)', 'Acceleration (m/s/s)', 'Alt Predicted'])
df_out = pd.DataFrame(out, columns=['Time (s)', 'Position (m)', 'Velocity (m/s)', 'Acceleration (m/s/s)', 'Alt Predicted'])

# Create separate plots for position, velocity, and acceleration, overlaying df and df_out, and saving to png
plt.figure(dpi=400) # Increase dpi for better quality
sns.set_theme(style='darkgrid')
sns.lineplot(data=df, x='Time (s)', y='Position (m)', label='Complementary Filter')
sns.lineplot(data=df_out, x='Time (s)', y='Position (m)', label='Kalman Filter')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.savefig('graphs/position.png')
plt.clf()

plt.figure(dpi=400) # Increase dpi for better quality
sns.set_theme(style='darkgrid')
sns.lineplot(data=df, x='Time (s)', y='Velocity (m/s)', label='Euler Integration')
sns.lineplot(data=df_out, x='Time (s)', y='Velocity (m/s)', label='Kalman Filter')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.savefig('graphs/velocity.png')
plt.clf()

plt.figure(dpi=400) # Increase dpi for better quality
sns.set_theme(style='darkgrid')
sns.lineplot(data=df, x='Time (s)', y='Acceleration (m/s/s)', label='Measurement')
sns.lineplot(data=df_out, x='Time (s)', y='Acceleration (m/s/s)', label='Kalman Filter')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s/s)')
plt.savefig('graphs/acceleration.png')
plt.clf()


"""
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
"""
