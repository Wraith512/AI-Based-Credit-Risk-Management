# %% [markdown]
# SENTIMENTAL ANALYSIS (using tweepy)
# 

# %%
#this is the code for sentimental analysis for twitter 
#using twitter api for realtime analysis (tweepy)

# %%
#importing the necesssary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer




# %%
#importing the dataset 
df=pd.read_csv(r"C:\Users\ASUS\Desktop\training datasets\twitter_training.csv\twitter_training.csv")

# %%
df.tail(5),df.head(5)

# %%
df.info()

# %%
df.isnull().sum()

#outlier detection
def detect_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers


numeric_col = df.select_dtypes(include=np.number).columns[0]

outliers = detect_outliers_iqr(df[numeric_col])

print("Column checked:", numeric_col)
print("Number of outliers:", len(outliers))
print(outliers.head())



