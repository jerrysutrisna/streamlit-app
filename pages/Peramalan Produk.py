import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Judul
st.title("Analisis dan Visualisasi Peramalan Produk")

st.markdown("""
**Permintaan Produk** merujuk pada jumlah barang atau unit tertentu yang diminta atau dibutuhkan oleh pelanggan dalam periode waktu tertentu. Fokusnya adalah pada pola permintaan aktual dari data historis yang tersedia, biasanya untuk satu jenis produk atau kelompok produk tertentu.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.style.hide(axis="index"))

    required_columns = {'Jumlah', 'Nama Barang', 'Tanggal'}
    if required_columns.issubset(df.columns):
        df['Jumlah'] = df['Jumlah'].astype(str).str.replace(r'[^\d]', '', regex=True)
        df['Jumlah'] = pd.to_numeric(df['Jumlah'], errors='coerce').fillna(0).astype(int)
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip().str.lower()
        df['Nama Barang'] = df['Nama Barang'].str.replace(r'\s+', ' ', regex=True)
        df = df[~df['Nama Barang'].str.contains('pekerjaan', case=False)]
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        df = df.dropna(subset=['Tanggal'])

        df_cleaned = df[df['Jumlah'] > 0]

        # Menampilkan 5 barang dengan jumlah unit terbanyak
        df_agg = df_cleaned.groupby('Nama Barang', as_index=False).agg({'Jumlah': 'sum'})
        top_items = df_agg.nlargest(5, 'Jumlah').reset_index(drop=True)

        st.write("5 Barang dengan Jumlah Unit Terbanyak:")
        top_display = top_items[['Nama Barang', 'Jumlah']].copy()
        top_display.insert(0, 'No', range(1, len(top_display) + 1))
        st.dataframe(top_display.style.hide(axis="index"))

        # Visualisasi 5 barang dengan jumlah unit terbanyak
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(x=top_items['Nama Barang'], y=top_items['Jumlah'], marker='o', linestyle='-', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 5 Barang dengan Jumlah Unit Terbanyak")
        plt.xlabel("Nama Barang")
        plt.ylabel("Jumlah")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Filter data untuk produk teratas
        top_products = df_cleaned[df_cleaned['Nama Barang'].isin(top_items['Nama Barang'])]
        top_products['Bulan'] = top_products['Tanggal'].dt.to_period('M')
        top_products_grouped = top_products.groupby(['Bulan', 'Nama Barang'])['Jumlah'].sum().reset_index()
        top_products_grouped['Bulan'] = pd.to_datetime(top_products_grouped['Bulan'].astype(str))

        # Visualisasi tren permintaan bulanan
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

        # Inisialisasi produk yang lolos dan tidak lolos ADF
        adf_pass_products = []
        adf_fail_products = []
        insufficient_data_products = []
        resampled_products = {}

        for product in df_cleaned['Nama Barang'].unique():
            product_data = df_cleaned[df_cleaned['Nama Barang'] == product].set_index('Tanggal')['Jumlah']
            product_data_resampled = product_data.resample('W').sum()

            if len(product_data_resampled.dropna()) > 10:
                result = adfuller(product_data_resampled.dropna())
                if result[1] <= 0.05:
                    adf_pass_products.append(product)
                    resampled_products[product] = product_data_resampled
                else:
                    adf_fail_products.append(product)
            else:
                insufficient_data_products.append(product)

        # Pilih metode peramalan
        st.subheader("Pilih Metode Peramalan")
        mode = st.radio("Metode:", ["Top 5 Produk Teratas", "Pilih Produk Sendiri"])

        selected_products = []
        if mode == "Top 5 Produk Teratas":
            selected_products = [p for p in top_items['Nama Barang'] if p in adf_pass_products]

            if not selected_products:
                st.warning("Tidak ada dari Top 5 produk yang lolos uji ADF. Silakan pilih produk sendiri.")

            # Tampilkan produk dari Top 5 yang gagal ADF atau kurang data
            gagal_adf = [p for p in top_items['Nama Barang'] if p in adf_fail_products]
            kurang_data = [p for p in top_items['Nama Barang'] if p in insufficient_data_products]

            if gagal_adf or kurang_data:
                st.subheader("Produk dari Top 5 yang Tidak Lolos Peramalan:")
                if gagal_adf:
                    st.write("Produk tidak lolos uji ADF:", ', '.join(gagal_adf))
                if kurang_data:
                    st.write("Produk dengan data tidak cukup:", ', '.join(kurang_data))

        else:
            if adf_pass_products:
                selected_product = st.selectbox("Pilih produk untuk dilakukan peramalan:", adf_pass_products)
                selected_products = [selected_product]
            else:
                st.warning("Tidak ada produk yang lolos uji ADF.")

        # Lakukan peramalan
        if selected_products and st.button("Lakukan Peramalan"):
            for product in selected_products:
                st.write(f"\n### Peramalan untuk {product}")
                product_data_resampled = resampled_products[product]

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                plot_acf(product_data_resampled.dropna(), ax=axes[0])
                axes[0].set_title(f"ACF {product} (Mingguan)")
                plot_pacf(product_data_resampled.dropna(), ax=axes[1])
                axes[1].set_title(f"PACF {product} (Mingguan)")
                st.pyplot(fig)

                model = SARIMAX(product_data_resampled,
                                order=(1, 0, 1),
                                seasonal_order=(1, 0, 1, 52),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit()

                forecast_object = results.get_forecast(steps=52)
                forecast_values = forecast_object.predicted_mean
                forecast_ci = forecast_object.conf_int()

                # Konversi index menjadi datetime jika perlu
                if not isinstance(forecast_values.index, pd.DatetimeIndex):
                    last_date = product_data_resampled.index[-1]
                    forecast_values.index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=52, freq='W')
                    forecast_ci.index = forecast_values.index

                # Mengatasi nilai prediksi negatif atau tidak realistis
                forecast_values = forecast_values.clip(lower=0)  # Membatasi nilai minimum menjadi 0
                forecast_ci = forecast_ci.clip(lower=0)  # Membatasi nilai minimum interval CI menjadi 0

                # Plot hasil prediksi
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(product_data_resampled, label="Data Aktual")
                ax.plot(forecast_values, label="Prediksi SARIMA", linestyle="dashed")
                ax.fill_between(forecast_values.index,
                                forecast_ci.iloc[:, 0],
                                forecast_ci.iloc[:, 1],
                                color='lightblue', alpha=0.4, label="Confidence Interval")
                ax.legend()
                ax.set_title(f"Prediksi SARIMA (12 Bulan ke Depan) untuk {product}")
                st.pyplot(fig)

                # Tabel hasil prediksi
                forecast_df = pd.DataFrame({
                    'Minggu': forecast_values.index,
                    'Prediksi Jumlah': np.round(forecast_values.values).astype(int)
                })

                # Bar chart prediksi
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Minggu', y='Prediksi Jumlah', data=forecast_df, palette='Blues_d', ax=ax)
                ax.set_xticks(ax.get_xticks()[::4])  # Hanya tampilkan sebagian label agar tidak rapat
                ax.set_xticklabels(forecast_df['Minggu'].dt.strftime('%Y-%m-%d')[::4], rotation=45)
                ax.set_title(f"Bar Chart Prediksi Jumlah per Minggu untuk {product}")
                ax.set_xlabel("Minggu")
                ax.set_ylabel("Jumlah")
                st.pyplot(fig)

                forecast_df_display = forecast_df.copy()
                forecast_df_display['Minggu'] = forecast_df_display['Minggu'].dt.strftime('%Y-%m-%d')
                forecast_df_display.insert(0, 'No', range(1, len(forecast_df_display) + 1))

                st.write(f"Tabel Hasil Prediksi SARIMA untuk {product}:")
                st.dataframe(forecast_df_display.style.hide(axis="index"))
