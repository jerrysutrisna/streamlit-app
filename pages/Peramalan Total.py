import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(page_title="Barang Forecasting", page_icon="📊")
st.title("Barang Forecasting with SARIMA")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Read Excel file
        input_data = pd.read_excel(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(input_data)

        # Check columns
        required_columns = ["Tanggal", "Jumlah"]
        if not all(col in input_data.columns for col in required_columns):
            st.error(f"Uploaded file must contain columns: {required_columns}")
            st.stop()

        # Clean and prepare
        input_data["Jumlah"] = input_data["Jumlah"].astype(str).str.replace(r'\D', '', regex=True).astype(float)
        input_data.dropna(inplace=True)
        input_data["Tanggal"] = pd.to_datetime(input_data["Tanggal"])
        input_data = input_data.sort_values("Tanggal")

        # Drop duplicate Tanggal
        input_data = input_data.drop_duplicates(subset="Tanggal")

        input_data.set_index("Tanggal", inplace=True)

        # Dominant year detection
        dominant_year = int(pd.Series(input_data.index.year).mode()[0])
        st.info(f"Data terbanyak berasal dari tahun: {dominant_year}")

        # Map model files
        model_map = {
            2022: "modelsarima2022.pkl",
            2023: "modelsarima2023.pkl",
            2024: "modelsarima2024.pkl"
        }

        # Tambahkan dominant_year ke list jika belum ada
        year_options = list(model_map.keys())
        if dominant_year not in year_options:
            year_options.append(dominant_year)

        # Allow user to override dominant year
        selected_year = st.selectbox(
            "Pilih tahun model SARIMA (override jika perlu):",
            options=sorted(year_options),
            index=year_options.index(dominant_year)
        )

        # Siapkan nama file model
        MODEL_FILE = model_map.get(selected_year, f"modelsarima{selected_year}.pkl")

         # Visualisasi bulanan
        st.subheader("📅 Jumlah Barang per Bulan")
        monthly_data = input_data["Jumlah"].resample("M").sum()
        st.dataframe(monthly_data.reset_index().rename(columns={"Jumlah": "Total Jumlah"}))
        st.bar_chart(monthly_data)
        st.line_chart(monthly_data)
        
        # ADF Test
        st.subheader("Uji Stasioneritas (ADF Test)")
        adf_result = adfuller(input_data["Jumlah"])
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write("Critical Values:")
        for key, value in adf_result[4].items():
            st.write(f"   {key}: {value}")
        if adf_result[1] > 0.05:
            st.warning("Data tidak stasioner, pertimbangkan differencing!")
        else:
            st.success("Data sudah stasioner!")

        # Plot ACF & PACF
        st.subheader("Plot ACF & PACF")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(input_data["Jumlah"], ax=axes[0])
        plot_pacf(input_data["Jumlah"], ax=axes[1])
        st.pyplot(fig)

        # Input parameter SARIMA
        st.subheader("Parameter SARIMA")
        p = st.number_input("Masukkan nilai p (AR)", min_value=0, value=1)
        d = st.number_input("Masukkan nilai d (Diff)", min_value=0, value=1)
        q = st.number_input("Masukkan nilai q (MA)", min_value=0, value=1)
        P = st.number_input("Masukkan nilai P (Seasonal AR)", min_value=0, value=1)
        D = st.number_input("Masukkan nilai D (Seasonal Diff)", min_value=0, value=1)
        Q = st.number_input("Masukkan nilai Q (Seasonal MA)", min_value=0, value=1)
        s = st.number_input("Masukkan nilai s (Seasonality)", min_value=1, value=12)

        # Load or train model
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                model_fit = pickle.load(f)
            st.success(f"Model {MODEL_FILE} berhasil dimuat.")
        else:
            st.warning(f"Model file untuk tahun {selected_year} ({MODEL_FILE}) tidak ditemukan.")
            st.info("Model akan dibuat dari data yang tersedia dengan parameter SARIMA yang Anda masukkan.")
            try:
                sarima_model = sm.tsa.statespace.SARIMAX(
                    input_data["Jumlah"],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False                         
                )
                model_fit = sarima_model.fit(disp=False)
                st.success("Model SARIMA berhasil dibuat.")

                # Save model
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(model_fit, f)
                st.success(f"Model baru berhasil disimpan sebagai {MODEL_FILE}.")

            except Exception as train_err:
                st.error(f"Gagal membuat model SARIMA: {train_err}")
                st.stop()

        # Forecast
        st.subheader("Peramalan")
        forecast_years = st.slider("Pilih jumlah tahun untuk peramalan:", 1, 5, 1)
        input_periods = forecast_years * 12

        if st.button("Predict"):
            forecast_result = model_fit.get_forecast(steps=input_periods)
            forecast_mean = forecast_result.predicted_mean
            last_date = input_data.index[-1]
            forecast_index = pd.date_range(start=last_date, periods=input_periods + 1, freq='M')[1:]
            forecast_df = pd.DataFrame({'Tanggal': forecast_index, 'Prediksi Jumlah': forecast_mean.values})

            st.subheader("📈 Hasil Peramalan")
            st.line_chart(forecast_df.set_index("Tanggal"))

            output_filename = "forecast_results.xlsx"
            forecast_df.to_excel(output_filename, index=False)
            with open(output_filename, "rb") as output_file:
                st.download_button("Download Forecast Results", output_file, file_name=output_filename)

            st.subheader("📊 Dashboard Visualisasi Hasil Prediksi")
            combined_df = pd.concat([
                input_data["Jumlah"].rename("Jumlah Aktual"),
                forecast_df.set_index("Tanggal")["Prediksi Jumlah"]
            ], axis=1)
            st.line_chart(combined_df)

            total_prediksi = forecast_df["Prediksi Jumlah"].sum()
            mean_prediksi = forecast_df["Prediksi Jumlah"].mean()
            growth_rate = ((forecast_df["Prediksi Jumlah"].iloc[-1] - forecast_df["Prediksi Jumlah"].iloc[0]) / forecast_df["Prediksi Jumlah"].iloc[0]) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Prediksi", f"{total_prediksi:,.0f}")
            col2.metric("Rata-rata / bulan", f"{mean_prediksi:,.2f}")
            col3.metric("Growth Rate", f"{growth_rate:.2f}%", delta=f"{growth_rate:.2f}%")

            st.subheader("📊 Bar Chart")
            st.bar_chart(forecast_df.set_index("Tanggal"))

            st.subheader("🌊 Area Chart")
            st.area_chart(forecast_df.set_index("Tanggal"))

            st.subheader("📋 Tabel Hasil Prediksi")
            st.dataframe(forecast_df.style.format({"Prediksi Jumlah": "{:,.2f}"}))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
