import streamlit as st
import pandas as pd
st.write("h1")

x = st.text_input("whats yourname")
st.write(f"hello{x}")

df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')
st.write(df)

# st.table(df)




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

st.write("Shape of X:", X.shape)
st.write("Shape of y:", y.shape)

# แบ่ง dataframe เป็น 2 ส่วนหลัก ๆ โดยมีส่วนที่ใช้สำหรับฝึกโมเดลและทดสอบโมเดล
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R squared: ", r2_score(Y_test, Y_pred))

# พล็อตกราฟเปรียบเทียบ
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Xcor')
plt.ylabel('Ycor')
plt.title('Predict')
plt.legend()
plt.grid(True)
st.pyplot(plt)
