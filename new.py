import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'sustainable_waste_management_dataset_2024.csv'
df = pd.read_csv(url, parse_dates=['date'])

df.head()


selected_features = ['population','overflow','waste_kg','organic_kg','collection_capacity_kg','temp_c','rain_mm']
X = df[selected_features]
y = df['recyclable_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['recyclable_kg']

# แบ่ง dataframe เป็น 2 ส่วนหลัก ๆ โดยมีส่วนที่ใช้สำหรับฝึกโมเดลและทดสอบโมเดล
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)




# พล็อตกราฟเปรียบเทียบ
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Xcor')
plt.ylabel('Ycor')
plt.title('Predict')
plt.legend()
plt.grid(True)


#แสดง ผลทางหน้าจอ 


st.write("Untitle Webside")

if 'show' not in st.session_state:
    st.session_state.show_ai = False

if st.button("Ai status"):
    st.session_state.show_ai = not st.session_state.show_ai

if st.session_state.show_ai:
    st.write("ความเฉลี่ยความคลาดเคลื่อน:", mean_squared_error(Y_test, Y_pred))
    st.write("R square: ", r2_score(Y_test, Y_pred))

if 'show' not in st.session_state:
    st.session_state.show_graph = False

if st.button("Linear graph"):
    st.session_state.show_graph = not st.session_state.show_graph

if st.session_state.show_graph:
    st.pyplot(plt)
