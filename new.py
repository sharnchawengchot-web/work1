import streamlit as st
import pandas as pd
st.write("h1")

x = st.text_input("whats yourname")
st.write(f"hello{x}")

df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')
st.write(df)

from numpy import *
# st.table(df)



'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = '/sustainable_waste_management_dataset_2024.csv'
df = pd.read_csv(url, parse_dates=['date'])

df.head()


selected_features = ['population','overflow','waste_kg','organic_kg','collection_capacity_kg','temp_c','rain_mm']
X = df[selected_features]
y = df['recyclable_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['recyclable_kg']

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
'''