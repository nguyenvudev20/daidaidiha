import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Amazon Rating Predictor", layout="wide")

st.title("🔮 Dự đoán đánh giá sản phẩm Amazon với Linear Regression")

# Tải dữ liệu trực tiếp từ GitHub
data_url = "https://raw.githubusercontent.com/nguyenvudev20/daidaidiha/refs/heads/main/amazon.csv"
st.info("📥 Dữ liệu đang được tải từ GitHub...")

try:
    df = pd.read_csv(data_url)

    st.subheader("📌 Dữ liệu gốc")
    st.dataframe(df.head())

    # Tiền xử lý dữ liệu
    df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
    df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
    df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = df['rating_count'].replace(',', '', regex=True)
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'], inplace=True)

    # Huấn luyện mô hình
    X = df[['actual_price', 'discounted_price', 'discount_percentage', 'rating_count']]
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success("✅ Mô hình huấn luyện hoàn tất!")
    st.metric("📉 Mean Squared Error", f"{mse:.4f}")
    st.metric("📈 R-squared Score", f"{r2:.4f}")

    # Biểu đồ so sánh dự đoán
    st.subheader("🔍 So sánh Rating thực tế vs. Dự đoán")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.6)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax1.set_xlabel("Giá trị thực tế (rating)")
    ax1.set_ylabel("Giá trị dự đoán")
    ax1.set_title("Rating thực tế vs. Rating dự đoán")
    st.pyplot(fig1)

    # Biểu đồ phân phối giá sau giảm
    st.subheader("📊 Phân phối giá sau giảm")
    fig2, ax2 = plt.subplots()
    df['discounted_price'].plot.hist(bins=30, ax=ax2)
    ax2.set_title("Histogram: Giá sau giảm")
    ax2.set_xlabel("Giá (VNĐ)")
    st.pyplot(fig2)

    # Biểu đồ scatter % giảm vs. rating
    st.subheader("📉 Mối quan hệ % giảm giá và đánh giá")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='discount_percentage', y='rating', ax=ax3)
    ax3.set_title("% Giảm giá vs. Rating")
    st.pyplot(fig3)

    # Biểu đồ giá trung bình theo danh mục
    st.subheader("💰 Giá trung bình theo danh mục chính")
    df['main_category'] = df['category'].str.split('|').str[0]
    avg_price_by_cat = df.groupby('main_category')['discounted_price'].mean().sort_values()
    fig4, ax4 = plt.subplots()
    avg_price_by_cat.plot(kind='barh', ax=ax4)
    ax4.set_xlabel("Giá trung bình (VNĐ)")
    ax4.set_title("Giá trung bình theo danh mục")
    st.pyplot(fig4)

    # Heatmap tương quan
    st.subheader("🧠 Ma trận tương quan giữa các biến")
    fig5, ax5 = plt.subplots()
    sns.heatmap(df[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']].corr(), annot=True, cmap="Blues", ax=ax5)
    st.pyplot(fig5)

    # Hệ số hồi quy
    coef_df = pd.DataFrame({
        'Đặc trưng': X.columns,
        'Hệ số (Coefficient)': model.coef_
    })
    st.subheader("📈 Ảnh hưởng của từng đặc trưng đến Rating")
    st.dataframe(coef_df)

except Exception as e:
    st.error(f"❌ Lỗi tải hoặc xử lý dữ liệu: {e}")
