import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

milk = pd.read_csv('monthly-milk-production.csv', index_col='Month')

milk.index = pd.to_datetime(milk.index)

# milk.plot()
# plt.show()

print(milk.info())
# Total of 168 entries
train_set = milk.head(156)
test_set = milk.tail(12)

# Scale Data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.fit(test_set)

# Batch function


def next_batch(training_data, batch_size, steps):
    # Random starting point
    rand_start = np.random.randint(0, len(training_data)-steps)

    # Create Y data for time series in batches
    y_batch = np.array(
        training_data[rand_start: rand_start + steps + 1]).reshape(1, steps+1)

    # Retunr the batches
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)
