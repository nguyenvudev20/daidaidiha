import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Amazon Rating Predictor", layout="wide")

# Title
st.title("🔮 Dự đoán đánh giá sản phẩm Amazon")

# Upload file
uploaded_file = st.file_uploader("📁 Tải lên file amazon.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Dữ liệu gốc")
    st.dataframe(df.head())

    # --- Tiền xử lý dữ liệu ---
    try:
        df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
        df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
        df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating_count'] = df['rating_count'].replace(',', '', regex=True)
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

        df.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'], inplace=True)

        # Feature và Target
        X = df[['actual_price', 'discounted_price', 'discount_percentage', 'rating_count']]
        y = df['rating']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Dự đoán
        y_pred = model.predict(X_test)

        # Hiển thị kết quả
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Mô hình huấn luyện hoàn tất!")
        st.metric("📉 Mean Squared Error", f"{mse:.4f}")
        st.metric("📈 R-squared Score", f"{r2:.4f}")

        # Biểu đồ so sánh dự đoán
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.set_xlabel("Giá trị thực tế (rating)")
        ax.set_ylabel("Giá trị dự đoán")
        ax.set_title("🔍 So sánh Rating thực tế vs. Dự đoán")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        st.pyplot(fig)

        # Hiển thị hệ số hồi quy
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        })
        st.subheader("📊 Ảnh hưởng của từng đặc trưng")
        st.dataframe(coef_df)

    except Exception as e:
        st.error(f"❌ Lỗi xử lý dữ liệu: {e}")

