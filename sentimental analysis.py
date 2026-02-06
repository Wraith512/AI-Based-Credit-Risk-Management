# %% [markdown]
# SENTIMENTAL ANALYSIS
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
df.tail(5)

# %%
df.info()

# %%
df.isnull().sum()


