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

# Fungsi untuk menambahkan CSS berdasarkan pilihan tema
def set_theme(theme_choice):
    if theme_choice == "Gelap":
        # CSS untuk tema gelap
        st.markdown(
            """
            <style>
            body {
                background-color: #121212;
                color: white;
            }
            .streamlit-expanderHeader {
                color: white;
            }
            .stButton>button {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #333;
            }
            .stTextInput>div>input {
                background-color: #2a2a2a;
                color: white;
            }
            .stSelectbox>div>div>input {
                background-color: #2a2a2a;
                color: white;
            }
            .stRadio>div>div>label {
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # CSS untuk tema terang
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: black;
            }
            .streamlit-expanderHeader {
                color: black;
            }
            .stButton>button {
                background-color: #f0f0f0;
                color: black;
                border: 1px solid #ccc;
            }
            .stTextInput>div>input {
                background-color: #ffffff;
                color: black;
            }
            .stSelectbox>div>div>input {
                background-color: #ffffff;
                color: black;
            }
            .stRadio>div>div>label {
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Sidebar untuk memilih tema
theme_choice = st.sidebar.radio("Pilih Tema", ("Terang", "Gelap"))

# Terapkan tema sesuai pilihan
set_theme(theme_choice)

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
1. **Unggah file Excel** yang berisi dua kolom utama:
   - Tanggal: tanggal transaksi atau pencatatan jumlah barang.
   - Jumlah: jumlah barang (boleh mengandung karakter non-angka, akan dibersihkan otomatis).
2. Data akan diproses, divisualisasikan, dan diuji stasioneritasnya.
3. Aplikasi menampilkan **plot ACF dan PACF** sebagai panduan parameter SARIMA.
4. Forecast akan dilakukan menggunakan model SARIMA yang telah dilatih sebelumnya.
5. Hasil prediksi akan ditampilkan dalam bentuk **tabel, grafik garis, batang, dan area**, serta dapat diunduh sebagai file Excel.

### 📁 Format Data Contoh:
| Tanggal     | Jumlah     |
|-------------|------------|
| 2022-01-01  | 1.000      |
| 2022-02-01  | 1.200      |
| 2022-03-01  | 900        |

> Pastikan format tanggal valid dan kolom sesuai agar proses berjalan lancar.
""")
