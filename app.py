import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib 
import plotly.graph_objs as go
from sklearn.cluster import KMeans

st.title("Analisis Segmentasi Pelanggan Mall")

st.sidebar.title("Menu")

option = st.sidebar.selectbox(
    "Pilih tampilan:",
    [
        "Deskripsi", 
        "Sumber Dataset", 
        "Nama Pembuat", 
        "Lihat Dataset", 
        "Visualisasi Data", 
        "Overview Algoritma", 
        "Hasil Clustering"
    ]
)

if option == "Deskripsi":
    st.write("""
    ### Deskripsi Proyek
    Aplikasi ini menggunakan algoritma **K-Means Clustering** untuk melakukan segmentasi pelanggan mall berdasarkan umur, penghasilan, dan skor pengeluaran.
    Dataset yang digunakan diambil dari [Kaggle](https://www.kaggle.com) yang berisi data pelanggan mall, termasuk usia, penghasilan tahunan, dan skor pengeluaran mereka.
    """)

if option == "Sumber Dataset":
    st.write("""
    ### Sumber Dataset
    Dataset ini diambil dari [Kaggle](https://www.kaggle.com/code/vitaaprilia/analysis-segmentation-customer-mall/input). Dataset berisi data pelanggan mall yang mencakup atribut seperti:
    - **Age**: Umur pelanggan
    - **Annual Income (k$)**: Penghasilan tahunan pelanggan dalam ribuan dolar
    - **Spending Score (1-100)**: Skor pengeluaran pelanggan (1-100)
    - **Gender**: Jenis kelamin pelanggan (Male/Female)
    """)

if option == "Nama Pembuat":
    st.write("""
    ### Nama Pembuat
    Proyek ini dibuat oleh: **Ranto**. Aplikasi ini bertujuan untuk memberikan wawasan tentang segmentasi pelanggan di mall menggunakan K-Means Clustering.
    """)

if option == "Lihat Dataset":
    df = pd.read_csv('Mall_Customers.csv', index_col=0)
    st.write(df)

if option == "Visualisasi Data":
    df = pd.read_csv('Mall_Customers.csv', index_col=0)
    df = df.rename(columns={'Annual Income (k$)': 'Annual Income', 'Spending Score (1-100)': 'Spending Score'})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    st.subheader("Visualisasi Data")
    col = st.selectbox("Pilih Kolom:", df.columns)
    st.bar_chart(df[col].value_counts())

if option == "Overview Algoritma":
    st.write("""
    ### Overview Algoritma KMeans Clustering
    KMeans Clustering adalah algoritma pembelajaran tidak terawasi (unsupervised learning) yang digunakan untuk mengelompokkan data berdasarkan kesamaan fitur-fitur tertentu. 
    Pada kasus ini, KMeans digunakan untuk mengelompokkan pelanggan mall berdasarkan tiga fitur: usia, penghasilan tahunan, dan skor pengeluaran. 
    Model ini mencari pola dalam data tanpa label sebelumnya dan membagi data ke dalam sejumlah cluster yang telah ditentukan (dalam hal ini, 6 cluster).
    """)

df = pd.read_csv('Mall_Customers.csv', index_col=0)
df = df.rename(columns={'Annual Income (k$)': 'Annual Income', 'Spending Score (1-100)': 'Spending Score'})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df[['Age', 'Annual Income', 'Spending Score']])

try:
    kmeans = joblib.load('kmeans_model.pkl') 
except FileNotFoundError:
    st.write("Model tidak ditemukan, melatih model baru...")
    
    kmeans = KMeans(n_clusters=6, init='k-means++', random_state=123)
    kmeans.fit(x_scaled)
    
    joblib.dump(kmeans, 'kmeans_model.pkl')
    st.write("Model baru telah dilatih dan disimpan.")

if option == "Hasil Clustering":
    df['Cluster'] = kmeans.predict(x_scaled)

    fig = go.Figure(data=go.Scatter3d(
        x=df['Age'],
        y=df['Spending Score'],
        z=df['Annual Income'],
        mode='markers',
        marker=dict(color=df['Cluster'], size=8, opacity=0.9)
    ))
    fig.update_layout(scene=dict(
        xaxis_title='Age',
        yaxis_title='Spending Score',
        zaxis_title='Annual Income'
    ))
    st.plotly_chart(fig)
    st.subheader("Hasil Clustering")
    st.write(df.groupby('Cluster').mean())

    st.write("""
    ### Kesimpulan
    Dari hasil clustering, dapat disimpulkan bahwa pengeluaran tertinggi terdapat pada cluster ke-2. Pada cluster ini, rata-rata pelanggan berusia sekitar 32 tahun, dengan pendapatan tahunan sebesar 86.5385 dan pengeluaran rata-rata sebesar 82.1288, yang merupakan nilai pengeluaran tertinggi dibandingkan cluster lainnya, diumur ini biasanya sudah berkeluarga dan memiliki satu anak.

    Pengeluaran tertinggi berikutnya terdapat pada cluster ke-6, dengan rata-rata pengeluaran sebesar 76.9167. Pelanggan pada cluster ini memiliki rentang usia sekitar 25 tahun, yang umumnya merupakan fresh graduate yang baru memulai karir di dunia kerja. 
    """)
