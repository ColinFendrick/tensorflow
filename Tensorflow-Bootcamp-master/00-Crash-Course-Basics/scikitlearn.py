import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = np.random.randint(0, 100, (10, 2))
scaler_model = MinMaxScaler()
scaler_model.fit(data)
print(
    scaler_model.transform(data)
)
newData = np.random.randint(0, 101, (50, 4))
dfCols = ['f1', 'f2', 'f3', 'label']
df = pd.DataFrame(data=newData, columns=dfCols)


# split data into features and test
X = df[['f1', 'f2', 'f3']]
y = df['label']

# Split with train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
