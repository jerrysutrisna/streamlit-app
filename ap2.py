import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from graphviz import Digraph
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Sidebar informasi
with st.sidebar:
    st.image("logo.png", width=150)
    st.markdown("""
    ### ℹ️ Tentang Aplikasi
    Aplikasi ini memprediksi **permintaan produk** berdasarkan data historis menggunakan model **SARIMA**.

    **Fitur:**
    - Visualisasi tren historis dan musiman.
    - Prediksi jumlah barang di masa depan.
    - Grafik dan tabel interaktif.
    """)

# Judul
st.title("Analisis dan Visuaalisasi Peramalan Permintaan Produk dengan SARIMA")

st.markdown("""
### 🧠 Tentang Model SARIMA:
SARIMA merupakan pengembangan dari model ARIMA yang mempertimbangkan pola musiman (**seasonality**) dalam data deret waktu. Model ini cocok digunakan pada data yang menunjukkan pola berulang secara periodik, seperti bulanan atau tahunan.

### 🎯 Tujuan Aplikasi:
- Menganalisis pola permintaan produk dari waktu ke waktu.
- Menyediakan informasi prediksi jumlah barang di masa mendatang.
- Membantu pengambilan keputusan dalam manajemen stok dan logistik.

### 🔄 Alur Kerja Aplikasi:
# Membuat flowchart dengan Graphviz
dot = Digraph()

dot.attr(rankdir='LR', bgcolor='white', style='filled', fillcolor='#F9FAFB')

# Node
dot.node("1", "📁 Unggah File Excel\n(Kolom: Tanggal & Jumlah)", shape='box', style='filled', fillcolor='#E0F7FA')
dot.node("2", "🧹 Pra-pemrosesan Data\n(pembersihan jumlah, konversi tanggal)", shape='box', style='filled', fillcolor='#E8F5E9')
dot.node("3", "📊 Visualisasi & Uji Stasioneritas", shape='box', style='filled', fillcolor='#FFF3E0')
dot.node("4", "🔁 Plot ACF & PACF\n(pemilihan parameter SARIMA)", shape='box', style='filled', fillcolor='#F3E5F5')
dot.node("5", "🤖 Pelatihan Model SARIMA", shape='box', style='filled', fillcolor='#EDE7F6')
dot.node("6", "📈 Tampilkan Hasil Prediksi\n(tabel, grafik, dan unduh Excel)", shape='box', style='filled', fillcolor='#FFFDE7')

# Edge
dot.edges([("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")])

# Tampilkan di Streamlit
st.graphviz_chart(dot)

### 📁 Format Data Contoh:
| Tanggal     | Jumlah     |
|-------------|------------|
| 2022-01-01  | 1.000      |
| 2022-02-01  | 1.200      |
| 2022-03-01  | 900        |

> Pastikan format tanggal valid dan kolom sesuai agar proses berjalan lancar.
""")
