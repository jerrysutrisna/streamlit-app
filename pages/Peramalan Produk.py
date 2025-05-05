import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.style.hide(axis="index"))  # Sembunyikan index bawaan

    required_columns = {'Jumlah', 'Nama Barang', 'Tanggal'}
    if required_columns.issubset(df.columns):
        df['Jumlah'] = df['Jumlah'].astype(str).str.replace(r'[^\d]', '', regex=True)
        df['Jumlah'] = pd.to_numeric(df['Jumlah'], errors='coerce').fillna(0).astype(int)
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip().str.lower()
        df['Nama Barang'] = df['Nama Barang'].str.replace(r'\s+', ' ', regex=True)
        df = df[~df['Nama Barang'].str.contains('pekerjaan')]
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        df_cleaned = df[df['Jumlah'] > 0]

        df_agg = df_cleaned.groupby('Nama Barang', as_index=False).agg({'Jumlah': 'sum'})
        top_items = df_agg.nlargest(5, 'Jumlah').reset_index(drop=True)

        st.write("5 Barang dengan Jumlah Unit Terbanyak:")
        top_display = top_items[['Nama Barang', 'Jumlah']].copy()
        top_display.insert(0, 'No', range(1, len(top_display) + 1))
        st.dataframe(top_display.style.hide(axis="index"))

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(x=top_items['Nama Barang'], y=top_items['Jumlah'], marker='o', linestyle='-', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 5 Barang dengan Jumlah Unit Terbanyak")
        plt.xlabel("Nama Barang")
        plt.ylabel("Jumlah")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        top_products = df_cleaned[df_cleaned['Nama Barang'].isin(top_items['Nama Barang'])]
        top_products['Bulan'] = top_products['Tanggal'].dt.to_period('M')
        top_products_grouped = top_products.groupby(['Bulan', 'Nama Barang'])['Jumlah'].sum().reset_index()
        top_products_grouped['Bulan'] = pd.to_datetime(top_products_grouped['Bulan'].astype(str))

        fig, ax = plt.subplots(figsize=(10, 5))
        for product in top_items['Nama Barang']:
            product_data = top_products_grouped[top_products_grouped['Nama Barang'] == product]
            ax.plot(product_data['Bulan'], product_data['Jumlah'], marker='o', label=product)

        plt.title("Tren Permintaan 5 Produk Teratas (Agregasi Bulanan)")
        plt.xlabel("Bulan")
        plt.ylabel("Jumlah")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        for product in top_items['Nama Barang']:
            product_data = top_products[top_products['Nama Barang'] == product].set_index('Tanggal')['Jumlah']
            product_data_resampled = product_data.resample('W').sum()

            st.write(f"\nUji ADF untuk {product} setelah resampling ke mingguan:")
            if len(product_data_resampled.dropna()) > 10:
                result = adfuller(product_data_resampled.dropna())
                st.write(f"ADF Statistic: {result[0]}")
                st.write(f"p-value: {result[1]}")

                if result[1] <= 0.05:
                    st.write("Data stasioner. Melanjutkan ke peramalan SARIMA.")

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    plot_acf(product_data_resampled.dropna(), ax=axes[0])
                    axes[0].set_title(f"ACF {product} (Mingguan)")
                    plot_pacf(product_data_resampled.dropna(), ax=axes[1])
                    axes[1].set_title(f"PACF {product} (Mingguan)")
                    st.pyplot(fig)

                    p, d, q = 1, 0, 1
                    P, D, Q, m = 1, 0, 1, 52

                    model = SARIMAX(product_data_resampled,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, m),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    results = model.fit()

                    forecast_object = results.get_forecast(steps=12)
                    forecast_values = forecast_object.predicted_mean
                    forecast_ci = forecast_object.conf_int()

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(product_data_resampled, label="Data Aktual")
                    ax.plot(forecast_values, label="Prediksi SARIMA", linestyle="dashed")
                    ax.fill_between(forecast_values.index,
                                    forecast_ci.iloc[:, 0].astype(int),
                                    forecast_ci.iloc[:, 1].astype(int),
                                    color='lightblue', alpha=0.4, label="Confidence Interval")
                    ax.legend()
                    ax.set_title(f"Prediksi SARIMA dengan Confidence Interval untuk {product}")
                    st.pyplot(fig)

                    forecast_df = pd.DataFrame({
                        'Minggu': forecast_values.index,
                        'Prediksi Jumlah': forecast_values.values.astype(int)
                    })

                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x='Minggu', y='Prediksi Jumlah', data=forecast_df, palette='Blues_d', ax=ax)
                    ax.set_xticklabels(forecast_df['Minggu'].dt.strftime('%Y-%m-%d'), rotation=45)
                    ax.set_title(f"Bar Chart Prediksi Jumlah per Minggu untuk {product}")
                    ax.set_xlabel("Minggu")
                    ax.set_ylabel("Jumlah")
                    st.pyplot(fig)

                    st.write(f"Tabel Hasil Prediksi SARIMA untuk {product}:")
                    forecast_df['Minggu'] = forecast_df['Minggu'].dt.strftime('%Y-%m-%d')
                    forecast_df_display = forecast_df.copy()
                    forecast_df_display.insert(0, 'No', range(1, len(forecast_df_display) + 1))
                    st.dataframe(forecast_df_display.style.hide(axis="index"))
                else:
                    st.write("Data tidak stasioner, peramalan tidak dilakukan.")
            else:
                st.write("Masih tidak cukup data untuk uji ADF.")
    else:
        st.error("Kolom 'Jumlah', 'Nama Barang', dan/atau 'Tanggal' tidak ditemukan dalam file yang diunggah.")
