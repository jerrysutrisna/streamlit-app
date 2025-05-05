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

# =============================
# 🔘 PILIH JENIS PERAMALAN
# =============================

st.markdown("### 🔍 Pilih Jenis Peramalan:")

col1, col2 = st.columns(2)

with col1:
    produk_clicked = st.button("📦 Peramalan Produk")

with col2:
    total_clicked = st.button("📊 Peramalan Total")

# Gunakan session_state agar pilihan tetap tersimpan
if produk_clicked:
    st.session_state['menu'] = 'produk'
elif total_clicked:
    st.session_state['menu'] = 'total'

# =============================
# 🔄 TAMPILKAN MENU SESUAI PILIHAN
# =============================
if 'menu' in st.session_state:
    if st.session_state['menu'] == 'produk':
        st.success("✅ Anda memilih: Peramalan Produk")
        st.markdown("Silakan lanjutkan dengan mengunggah data produk untuk diprediksi.")
        # 👉 Tambahkan modul peramalan produk di sini

    elif st.session_state['menu'] == 'total':
        st.success("✅ Anda memilih: Peramalan Total")
        st.markdown("Silakan lanjutkan dengan mengunggah data total permintaan untuk diprediksi.")
        # 👉 Tambahkan modul peramalan total di sini
