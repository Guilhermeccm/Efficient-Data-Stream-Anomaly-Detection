import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time

# Constants
WINDOW_SIZE = 30  # Window size
Z_THRESHOLD = 2  # Threshold for anomaly detection
MAD_THRESHOLD = 3.0  # Median Absolute Deviation threshold for anomaly detection

# Creates a data stream generator
def data_stream():
    t = 0
    while True:
        # Simulates regular pattern: sine wave
        pattern = 10 * np.sin(0.2 * t)
        # Add random noise
        noise = np.random.normal(0, 1)
        # Occasionally adds an anomaly
        if random.random() < 0.02:  # 2% chance of an anomaly
            value = random.uniform(20, 30)
        else:
            value = pattern + noise
        t += 1
        yield value

# Z-Score based anomaly detection
def detect_anomaly_zscore(value, window):
    if len(window) < WINDOW_SIZE:
        return False, 0  # Not enough data points yet

    mean = np.mean(window)
    std_dev = np.std(window)
    if std_dev == 0:
        return False, 0  # Avoid division by zero
    z_score = (value - mean) / std_dev

    if abs(z_score) > Z_THRESHOLD:
        return True, z_score
    return False, z_score

# MAD based anomaly detection
def detect_anomaly_mad(value, window):
    if len(window) < WINDOW_SIZE:
        return False, 0  # Not enough data points yet

    median = np.median(window) # Median
    mad = np.median(np.abs(window - median))  # Median Absolute Deviation

    if mad == 0:
        return False, 0  # Avoid division by zero
    mad_score = np.abs(value - median) / mad

    if mad_score > MAD_THRESHOLD:
        return True, mad_score
    return False, mad_score

# Real-time data visualization
def plot_stream():
    stream = data_stream()
    window = deque(maxlen=WINDOW_SIZE)
    x_data, y_data = [], []
    anomaly_points = []

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    ax.set_ylim(-15, 35)

    while True:
        value = next(stream)
        window.append(value)
        
        # Checks for anomalies using Z-Score and MAD
        is_anomaly_z, z_score = detect_anomaly_zscore(value, window)
        is_anomaly_mad, mad_score = detect_anomaly_mad(value, window)

        # If anomaly is detected by either method, flags it
        if is_anomaly_z or is_anomaly_mad:
            anomaly_points.append((len(x_data), value))

        # Updates data for plotting
        x_data.append(len(x_data))
        y_data.append(value)
        
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()

        # Plots anomaly points
        if anomaly_points:
            ax.scatter(*zip(*anomaly_points), color='r')

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.02)  # Simulates delay for real-time streaming



if __name__ == "__main__":
    plot_stream()
