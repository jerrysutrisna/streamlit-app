import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

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

        year_options = list(model_map.keys())
        if dominant_year not in year_options:
            year_options.append(dominant_year)

        selected_year = st.selectbox(
            "Pilih tahun model SARIMA (override jika perlu):",
            options=sorted(year_options),
            index=year_options.index(dominant_year)
        )

        MODEL_FILE = model_map.get(selected_year, f"modelsarima{selected_year}.pkl")

        # Visualisasi bulanan
        st.subheader("📅 Jumlah Barang per Bulan")
        monthly_data = input_data["Jumlah"].resample("M").sum()
        monthly_df = monthly_data.reset_index().rename(columns={"Jumlah": "Total Jumlah"})

        fig_monthly = px.line(
            monthly_df,
            x="Tanggal",
            y="Total Jumlah",
            title="Total Jumlah Barang per Bulan",
            markers=True,
            hover_data={"Tanggal": "|%B %Y", "Total Jumlah": ":,.0f"}
        )
        fig_monthly.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah")
        st.plotly_chart(fig_monthly, use_container_width=True)

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

        # Plot ACF & PACF Interaktif
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

            forecast_df = pd.DataFrame({
                'Tanggal': forecast_index,
                'Prediksi Jumlah': np.round(forecast_mean.values).astype(int)
            })

            st.subheader("📈 Hasil Peramalan")
            fig_forecast = px.line(
                forecast_df,
                x="Tanggal",
                y="Prediksi Jumlah",
                title="Hasil Prediksi Jumlah Barang",
                markers=True,
                hover_data={"Tanggal": "|%B %Y", "Prediksi Jumlah": ":,.0f"}
            )
            fig_forecast.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Prediksi")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Download forecast
            output_filename = "forecast_results.xlsx"
            forecast_df.to_excel(output_filename, index=False)
            with open(output_filename, "rb") as output_file:
                st.download_button("Download Forecast Results", output_file, file_name=output_filename)

            # Combine actual and forecast
            st.subheader("📊 Dashboard Visualisasi Hasil Prediksi")
            combined_df = pd.concat([
                input_data["Jumlah"].rename("Jumlah Aktual"),
                forecast_df.set_index("Tanggal")["Prediksi Jumlah"]
            ], axis=1).reset_index()

            fig_combined = px.line(
                combined_df,
                x="Tanggal",
                y=["Jumlah Aktual", "Prediksi Jumlah"],
                title="Aktual vs Prediksi Jumlah Barang",
                markers=True
            )
            fig_combined.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah")
            st.plotly_chart(fig_combined, use_container_width=True)

            # KPI Metrics
            total_prediksi = forecast_df["Prediksi Jumlah"].sum()
            mean_prediksi = forecast_df["Prediksi Jumlah"].mean()
            growth_rate = ((forecast_df["Prediksi Jumlah"].iloc[-1] - forecast_df["Prediksi Jumlah"].iloc[0]) / forecast_df["Prediksi Jumlah"].iloc[0]) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Prediksi", f"{total_prediksi:,.0f}")
            col2.metric("Rata-rata / bulan", f"{mean_prediksi:,.0f}")
            col3.metric("Growth Rate", f"{growth_rate:.2f}%", delta=f"{growth_rate:.2f}%")

            # Bar Chart
            st.subheader("📊 Bar Chart")
            fig_bar = px.bar(forecast_df, x="Tanggal", y="Prediksi Jumlah", title="Bar Chart Prediksi")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Area Chart
            st.subheader("🌊 Area Chart")
            fig_area = px.area(forecast_df, x="Tanggal", y="Prediksi Jumlah", title="Area Chart Prediksi")
            st.plotly_chart(fig_area, use_container_width=True)

            # Table
            st.subheader("📋 Tabel Hasil Prediksi")
            st.dataframe(forecast_df.style.format({"Prediksi Jumlah": "{:,.0f}"}))

            # Distribusi Prediksi Bulanan
            st.subheader("Status Prediksi Bulanan")
            lead_status_fig = px.bar(
                forecast_df,
                x=forecast_df["Tanggal"].dt.strftime("%b %Y"),
                y="Prediksi Jumlah",
                title="Distribusi Prediksi per Bulan"
            )
            st.plotly_chart(lead_status_fig, use_container_width=True)

            # Trend Prediksi Barang
            st.subheader("Trend Prediksi Barang per Bulan")
            fig_campaign = px.line(forecast_df, x="Tanggal", y="Prediksi Jumlah", title="Trend Prediksi per Bulan")
            st.plotly_chart(fig_campaign, use_container_width=True)

            # Pendapatan Tahunan (Akumulasi)
            forecast_df["Tahun"] = forecast_df["Tanggal"].dt.year
            yearly_income = forecast_df.groupby("Tahun")["Prediksi Jumlah"].sum().reset_index()

            st.subheader("Pendapatan Prediksi Tahunan")
            fig_year = px.line(yearly_income, x="Tahun", y="Prediksi Jumlah", markers=True, title="Pendapatan Prediksi Tahunan")
            st.plotly_chart(fig_year, use_container_width=True)

            # Distribusi Kuartal
            st.subheader("Distribusi Prediksi per Kuartal")
            forecast_df["Kuartal"] = forecast_df["Tanggal"].dt.to_period("Q").astype(str)
            kuartal_summary = forecast_df.groupby("Kuartal")["Prediksi Jumlah"].sum().reset_index()

            fig_kuartal = px.pie(
                kuartal_summary,
                names="Kuartal",
                values="Prediksi Jumlah",
                title="Distribusi Prediksi per Kuartal",
                hole=0.4
            )
            st.plotly_chart(fig_kuartal, use_container_width=True)

            # Proporsi Tahun (jika multi-year)
            if forecast_years > 1:
                st.subheader("Proporsi Prediksi per Tahun")
                fig_proporsi = px.pie(
                    yearly_income,
                    names="Tahun",
                    values="Prediksi Jumlah",
                    title="Proporsi Prediksi per Tahun",
                    hole=0.3
                )
                st.plotly_chart(fig_proporsi, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
