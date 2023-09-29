#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Introduce missing data by replacing some values in the "sepal_length" column with NaN
np.random.seed(0)
missing_indices = np.random.choice(iris_df.index, size=10, replace=False)
iris_df.loc[missing_indices, 'sepal length (cm)'] = np.nan


# In[2]:


# Impute missing values in the "sepal length (cm)" column with the mean value
mean_sepal_length = iris_df['sepal length (cm)'].mean()
iris_df['sepal length (cm)'].fillna(mean_sepal_length, inplace=True)


# In[3]:


# Calculate the average sepal length before handling missing data
average_sepal_length_before = iris_df['sepal length (cm)'].mean()
print(f"Average Sepal Length Before Handling Missing Data: {average_sepal_length_before:.2f}")

# Calculate the average sepal length after handling missing data
average_sepal_length_after = iris_df['sepal length (cm)'].mean()
print(f"Average Sepal Length After Handling Missing Data: {average_sepal_length_after:.2f}")


# In[ ]:




