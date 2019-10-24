import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('pima-indians-diabetes.csv')
print(diabetes.head())

# Cleaning data
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
                'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

print(diabetes.head())

# Feature Columns

# CONTINUOUS FEATURES
# Number of times pregnant
# Plasma glucose concentration at 2 hours in an oral glucose tolerance test
# Diastolic blood pressure(mm Hg)
# Triceps skin fold thickness(mm)
# 2-Hour serum insulin(mu U/ml)
# Body mass index(weight in kg/(height in m) ^ 2)
# Diabetes pedigree function

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# CATEGORICAL FEATURES
# If you know the set of all possible feature values of a column
# and there are only a few of them,
# you can use categorical_column_with_vocabulary_list.
# If you don't know the set of possible values in advance
# you can use categorical_column_with_hash_bucket.
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list(
    'Group', ['A', 'B', 'C', 'D'])

diabetes['Age'].hist(bins=20)
plt.show()

age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[20, 30, 40, 50, 60, 70, 80])
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin,
             bmi, diabetes_pedigree, assigned_group, age_buckets]

x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']
X_train, X_test, y_train, y_test = train_test_split(
    x_data, labels, test_size=0.33, random_state=101)

#  Input function
input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
