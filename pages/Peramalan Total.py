import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Barang Forecasting", page_icon="\ud83d\udcca")
st.title("Barang Forecasting with SARIMA (Refactor Version)")

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
        input_data["Jumlah"] = input_data["Jumlah"].astype(str).str.replace(r'\\D', '', regex=True).astype(float)
        input_data.dropna(inplace=True)
        input_data["Tanggal"] = pd.to_datetime(input_data["Tanggal"])
        input_data = input_data.sort_values("Tanggal").drop_duplicates(subset="Tanggal")
        input_data.set_index("Tanggal", inplace=True)

        # Dominant year detection
        dominant_year = int(input_data.index.year.mode()[0])
        st.info(f"Data terbanyak berasal dari tahun: {dominant_year}")

        # Model file mapping
        model_map = {2022: "modelsarima2022.pkl", 2023: "modelsarima2023.pkl", 2024: "modelsarima2024.pkl"}
        MODEL_FILE = model_map.get(dominant_year, f"modelsarima{dominant_year}.pkl")

        # Visualisasi Aktual
        st.header("\ud83d\udcc5 Visualisasi Data Aktual")
        monthly_data = input_data["Jumlah"].resample("M").sum()
        st.subheader("Total Barang per Bulan")
        fig_actual = px.line(monthly_data, labels={"value":"Jumlah", "Tanggal":"Bulan"}, title="Data Aktual per Bulan")
        st.plotly_chart(fig_actual, use_container_width=True)

        # Uji Stasioneritas
        st.header("\ud83d\udd22 Uji Stasioneritas (ADF Test)")
        adf_result = adfuller(input_data["Jumlah"])
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        if adf_result[1] > 0.05:
            st.warning("Data tidak stasioner, pertimbangkan differencing!")
        else:
            st.success("Data sudah stasioner!")

        # Plot ACF & PACF
        st.subheader("Plot ACF dan PACF")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(input_data["Jumlah"], ax=axes[0])
        plot_pacf(input_data["Jumlah"], ax=axes[1])
        st.pyplot(fig)

        # Parameter SARIMA
        st.header("\ud83d\udd22 Parameter SARIMA")
        p = st.number_input("p (AR)", min_value=0, value=1)
        d = st.number_input("d (Differencing)", min_value=0, value=1)
        q = st.number_input("q (MA)", min_value=0, value=1)
        P = st.number_input("P (Seasonal AR)", min_value=0, value=1)
        D = st.number_input("D (Seasonal Diff)", min_value=0, value=1)
        Q = st.number_input("Q (Seasonal MA)", min_value=0, value=1)
        s = st.number_input("Seasonality (s)", min_value=1, value=12)

        # Load or Train Model
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                model_fit = pickle.load(f)
            st.success(f"Model {MODEL_FILE} berhasil dimuat.")
        else:
            st.warning(f"Model {MODEL_FILE} tidak ditemukan. Membuat model baru...")
            model = sm.tsa.statespace.SARIMAX(input_data["Jumlah"],
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, s),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(model_fit, f)
            st.success(f"Model baru disimpan sebagai {MODEL_FILE}")

        # Forecast
        st.header("\ud83d\udcca Peramalan")
        forecast_years = st.slider("Jumlah Tahun Prediksi", 1, 5, 1)
        periods = forecast_years * 12

        if st.button("\ud83d\udca1 Mulai Prediksi"):
            forecast = model_fit.get_forecast(steps=periods)
            pred_mean = forecast.predicted_mean
            pred_index = pd.date_range(start=input_data.index[-1], periods=periods+1, freq='M')[1:]
            forecast_df = pd.DataFrame({"Tanggal": pred_index, "Prediksi Jumlah": pred_mean.values})

            # Visualisasi Prediksi
            st.subheader("\ud83d\udcc8 Hasil Prediksi")
            fig_pred = px.line(forecast_df, x="Tanggal", y="Prediksi Jumlah", title="Prediksi Barang per Bulan")
            st.plotly_chart(fig_pred, use_container_width=True)

            st.subheader("\ud83d\udcc8 Gabungan Aktual & Prediksi")
            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(x=input_data.index, y=input_data["Jumlah"], name="Aktual", mode="lines"))
            fig_combined.add_trace(go.Scatter(x=forecast_df["Tanggal"], y=forecast_df["Prediksi Jumlah"], name="Prediksi", mode="lines"))
            fig_combined.update_layout(title="Data Aktual vs Prediksi")
            st.plotly_chart(fig_combined, use_container_width=True)

            # Dashboard Prediksi
            st.header("\ud83d\udcca Ringkasan Prediksi")
            total = forecast_df["Prediksi Jumlah"].sum()
            mean = forecast_df["Prediksi Jumlah"].mean()
            growth = ((forecast_df["Prediksi Jumlah"].iloc[-1] - forecast_df["Prediksi Jumlah"].iloc[0]) / forecast_df["Prediksi Jumlah"].iloc[0]) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Prediksi", f"{total:,.0f}")
            col2.metric("Rata-rata per Bulan", f"{mean:,.2f}")
            col3.metric("Growth Rate", f"{growth:.2f}%")

            st.subheader("Distribusi Prediksi per Tahun")
            forecast_df["Tahun"] = forecast_df["Tanggal"].dt.year
            yearly_summary = forecast_df.groupby("Tahun")["Prediksi Jumlah"].sum().reset_index()
            fig_yearly = px.bar(yearly_summary, x="Tahun", y="Prediksi Jumlah", title="Total Prediksi per Tahun")
            st.plotly_chart(fig_yearly, use_container_width=True)

            st.subheader("Distribusi Prediksi per Kuartal")
            forecast_df["Kuartal"] = forecast_df["Tanggal"].dt.to_period("Q").astype(str)
            quarterly_summary = forecast_df.groupby("Kuartal")["Prediksi Jumlah"].sum().reset_index()
            fig_quarter = px.pie(quarterly_summary, names="Kuartal", values="Prediksi Jumlah", hole=0.4, title="Distribusi per Kuartal")
            st.plotly_chart(fig_quarter, use_container_width=True)

            # Download Forecast
            output_filename = "forecast_results.xlsx"
            forecast_df.to_excel(output_filename, index=False)
            with open(output_filename, "rb") as f:
                st.download_button("\ud83d\udcbe Download Forecast", f, file_name=output_filename)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
