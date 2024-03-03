# -*- coding: utf-8 -*-
"""
Normalization and standardization in Python
Spyder version 5.3.3
"""

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prepro
import numpy as np
# Let's look at a list of observations with different ranges

data_frame = pd.DataFrame( 
    [[3], [4], [65], [213], [4892], [3], [1.5], [3], [65]],
    columns = ['data'])

# See the profiel of the data
data_frame.plot(kind = 'bar')
data = np.array([3, 4, 65, 213, 4892, 3, 1.5, 3, 65])
# See how some values are outliers, more so 4892
# Normalization and Standarization help us pre-processing the data
# We actuslly have a built-in method in sklearn
normalized_data = prepro.normalize([data_frame['data'].values], norm = 'max')

# Our data is now normalized by using the min max normalization
# However, we can also do it ourselves with math
manual_normalized_data = (data_frame['data'] - data_frame['data'].min())/ (data_frame['data'].max() - data_frame['data'].min())

# If we want to match the formula above, we will need to use MinMaxScaler
scaler = prepro.MinMaxScaler()
min_max_normalized_data = scaler.fit_transform(data_frame.data.values.reshape(-1, 1))

# We can see that the result is the same as our formula
# Another way to pre-process data is called standardization
# Instead of looking at the minimum and maximum values, we end up with a distribution with a mean of 0 and a deviation of 1
manual_standarized_data = (data_frame['data'] - data_frame['data'].mean()) / data_frame['data'].std(ddof = 0)

# This can also be performed with skllearn
standard_scaler = prepro.StandardScaler().fit(data_frame.data.values.reshape(-1, 1))
scaled_data = standard_scaler.transform(data_frame.data.values.reshape(-1, 1))

# Normalization and standardization are good practices for machine learning models, even if these models do not require it
# Some models scale the data beforehand as well, or provide an option to do it