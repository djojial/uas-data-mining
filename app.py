# app.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ======================================
# Konfigurasi Halaman
# ======================================
st.set_page_config(page_title="UAS Data Mining", layout="wide")

st.sidebar.title("Menu Navigasi")
page = st.sidebar.radio(
    "Pilih Fitur",
    ["Klasifikasi Diabetes", "Clustering Gerai Kopi"]
)

st.title("Ujian Akhir Semester")
st.write("Nama: Muhammad Djoji Alamni")
st.write("NIM: 22146052")

# ======================================
# Klasifikasi Diabetes
# ======================================
if page == "Classification":
    st.subheader("Classification Diabetes")
    st.write("Prediksi kemungkinan diabetes menggunakan model KNN.")

    # Load dataset contoh
    data = pd.read_csv("diabetes.csv")
    st.dataframe(data.head())

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Load model KNN dari .pkl
    with open("knn_diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Evaluasi cepat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = model.predict(X_test)

    st.write("**Metrik Klasifikasi**")
    st.text(classification_report(y_test, y_pred))

    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)

    # Form prediksi
    st.write("**Input Data Baru**")
    col1, col2 = st.columns(2)
    input_data = {}
    for idx, col in enumerate(X.columns):
        if idx % 2 == 0:
            val = col1.number_input(f"{col}", value=float(X[col].mean()))
        else:
            val = col2.number_input(f"{col}", value=float(X[col].mean()))
        input_data[col] = val

    input_df = pd.DataFrame([input_data])

    if st.button("Prediksi Diabetes"):
        hasil = model.predict(input_df)
        st.success(f"Hasil Prediksi: {'Positif Diabetes' if hasil[0] == 1 else 'Negatif Diabetes'}")

# ======================================
# Clustering Gerai Kopi
# ======================================
elif page == "Clustering":
    st.subheader("Clustering Lokasi Gerai Kopi")
    st.write("Segmentasi lokasi gerai kopi dengan KMeans.")

    # Load dataset contoh
    data = pd.read_csv("lokasi_gerai_kopi_clean.csv")
    st.dataframe(data.head())

    X = data.select_dtypes(include=["float64", "int64"])

    # Load model KMeans dari .pkl
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = joblib.load(f)

    # Prediksi cluster
    clusters = kmeans.predict(X)
    data["Cluster"] = clusters
    st.write("**Data dengan Cluster:**")
    st.dataframe(data.head())

    # Scatter plot kalau ada 2 kolom numerik
    if X.shape[1] >= 2:
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap="viridis")
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_title("Visualisasi Clustering")
        st.pyplot(fig)
    else:
        st.info("Data kurang dari 2 kolom numerik untuk scatter plot.")

    # Form prediksi cluster baru
    st.write("**Input Data Baru**")
    col1, col2 = st.columns(2)
    input_vals = []
    for idx, col in enumerate(X.columns):
        if idx % 2 == 0:
            val = col1.number_input(f"{col}", value=float(X[col].mean()))
        else:
            val = col2.number_input(f"{col}", value=float(X[col].mean()))
        input_vals.append(val)

    if st.button("Prediksi Cluster"):
        hasil = kmeans.predict([input_vals])
        st.success(f"Data baru diprediksi masuk Cluster: {hasil[0]}")
