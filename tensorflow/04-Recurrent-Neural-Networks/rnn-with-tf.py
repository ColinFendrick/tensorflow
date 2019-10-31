import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData():

    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):
        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)
        # Convert to be on time series
        ts_start = rand_start * \
            (self.xmax - self.xmin - (steps*self.resolution))
        # Create batch Time Series on t axis
        batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution
        # Create Y data for time series in the batches
        y_batch = np.sin(batch_ts)

        # Format for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
        else:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


ts_data = TimeSeriesData(250, 0, 10)
plt.plot(ts_data.x_data, ts_data.y_true)
plt.show()

# Num of steps in batch
num_time_steps = 30

y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)

plt.plot(ts.flatten()[1:], y2.flatten(), '*')
plt.show()

plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='Single Training Instance')
plt.legend()
plt.tight_layout()
plt.show()
