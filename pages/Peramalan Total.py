import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import gzip
from statsmodels.tsa.stattools import adfuller

st.title("Analisis dan Visualisasi Peramalan Total")

st.markdown("""
**Permintaan Total (SARIMA)** adalah hasil dari proses peramalan berdasarkan model SARIMA, yang memproyeksikan jumlah permintaan ke masa depan dengan mempertimbangkan tren, musim (seasonality), dan fluktuasi historis dalam data. Permintaan total ini mencerminkan estimasi dari **seluruh permintaan** yang mungkin terjadi berdasarkan pola masa lalu, bukan hanya angka aktual dari data yang sudah terjadi.
""")

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
        # Tambahkan transformasi log untuk stabilisasi variansi
        input_data["Jumlah_Log"] = np.log1p(input_data["Jumlah"])
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
            2022: "modelsarima2022.pkl.gz",
            2023: "modelsarima2023.pkl.gz",
            2024: "modelsarima2024.pkl.gz"
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
        fig_monthly.update_layout(
            xaxis_title="Bulan",
            yaxis_title="Jumlah",
            hovermode="x unified",
            dragmode="select",
            selectdirection="h"
        )
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
        acf_values = sm.tsa.stattools.acf(input_data["Jumlah"], nlags=40)
        pacf_values = sm.tsa.stattools.pacf(input_data["Jumlah"], nlags=40)

        acf_fig = px.bar(
            x=list(range(len(acf_values))),
            y=acf_values,
            labels={'x': 'Lag', 'y': 'ACF'},
            title="Autocorrelation (ACF)"
        )
        acf_fig.update_layout(
            hovermode="x unified",
            dragmode="select",
            selectdirection="h",
            showlegend=False
        )

        pacf_fig = px.bar(
            x=list(range(len(pacf_values))),
            y=pacf_values,
            labels={'x': 'Lag', 'y': 'PACF'},
            title="Partial Autocorrelation (PACF)"
        )
        pacf_fig.update_layout(
            hovermode="x unified",
            dragmode="select",
            selectdirection="h",
            showlegend=False
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(acf_fig, use_container_width=True)
        with col2:
            st.plotly_chart(pacf_fig, use_container_width=True)

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
            with gzip.open(MODEL_FILE, "rb") as f:
                model_fit = pickle.load(f)
            st.success(f"Model {MODEL_FILE} berhasil dimuat.")
        else:
            st.warning(f"Model file untuk tahun {selected_year} ({MODEL_FILE}) tidak ditemukan.")
            st.info("Model akan dibuat dari data yang tersedia dengan parameter SARIMA yang Anda masukkan.")
            try:
                sarima_model = sm.tsa.statespace.SARIMAX(
                    input_data["Jumlah_Log"],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False                         
                )
                model_fit = sarima_model.fit(disp=False)
                st.success("Model SARIMA berhasil dibuat.")
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
            forecast_mean_log = forecast_result.predicted_mean
            forecast_mean = np.expm1(forecast_mean_log)  # Kembali ke skala jumlah barang
            last_date = input_data.index[-1]
            forecast_index = pd.date_range(start=last_date, periods=input_periods + 1, freq='M')[1:]

            forecast_df = pd.DataFrame({
                'Tanggal': forecast_index,
                'Prediksi_Log': forecast_mean_log.values,
                'Prediksi Jumlah': np.round(forecast_mean.values).astype(int)
            })

            st.dataframe(forecast_df)

            st.subheader("📈 Hasil Peramalan")
            fig_forecast = px.line(
                forecast_df,
                x="Tanggal",
                y="Prediksi Jumlah",
                title="Hasil Prediksi Jumlah Barang",
                markers=True,
                hover_data={"Tanggal": "|%B %Y", "Prediksi Jumlah": ":,.0f"}
            )
            fig_forecast.update_layout(
                xaxis_title="Tanggal",
                yaxis_title="Jumlah Prediksi",
                hovermode="x unified",
                dragmode="select",
                selectdirection="h"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Download forecast
            output_filename = "forecast_results.xlsx"
            forecast_df.to_excel(output_filename, index=False)
            with open(output_filename, "rb") as output_file:
                st.download_button("Download Forecast Results", output_file, file_name=output_filename)

            # Dashboard Prediksi dalam Skala Log
            st.subheader("📊 Dashboard Visualisasi Hasil Prediksi (Log)")

            # Gabungkan data aktual dan prediksi (log)
            combined_log_df = pd.concat([
                input_data["Jumlah_Log"].rename("Log Aktual"),
                forecast_df.set_index("Tanggal")["Prediksi_Log"].rename("Log Prediksi")
            ], axis=1).reset_index()

            # Plot log aktual vs log prediksi
            fig_combined_log = px.line(
                combined_log_df,
                x="Tanggal",
                y=["Log Aktual", "Log Prediksi"],
                title="Aktual vs Prediksi (Log) Jumlah Barang",
                markers=True
            )
            fig_combined_log.update_layout(
                xaxis_title="Tanggal",
                yaxis_title="Log Jumlah",
                hovermode="x unified",
                dragmode="select",
                selectdirection="h"
            )
            st.plotly_chart(fig_combined_log, use_container_width=True)

            # KPI Metrics
            total_prediksi = forecast_df["Prediksi Jumlah"].sum()
            mean_prediksi = forecast_df["Prediksi Jumlah"].mean()
            awal = forecast_df["Prediksi Jumlah"].iloc[0]
            akhir = forecast_df["Prediksi Jumlah"].iloc[-1]

            # Ambil nilai awal dan akhir prediksi yang bukan nol
            non_zero_preds = forecast_df[forecast_df["Prediksi Jumlah"] > 0]
            if not non_zero_preds.empty:
                awal = non_zero_preds["Prediksi Jumlah"].iloc[0]
                akhir = non_zero_preds["Prediksi Jumlah"].iloc[-1]
                growth_rate = ((akhir - awal) / awal) * 100
                delta_label = f"{growth_rate:.2f}%"
                growth_display = f"{growth_rate:.2f}%"
            else:
                awal = akhir = growth_rate = 0
                delta_label = "N/A"
                growth_display = "N/A"

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Prediksi", f"{total_prediksi:,.0f}")
            col2.metric("Rata-rata / bulan", f"{mean_prediksi:,.0f}")
            col3.metric(
                label="Growth Rate",
                value=f"{growth_rate:.2f}%" if awal != 0 else "N/A",
                delta=delta_label,
                delta_color="normal" if growth_rate == 0 else ("inverse" if growth_rate < 0 else "off")
            )

            # Bar Chart
            st.subheader("📊 Bar Chart")
            fig_bar = px.bar(forecast_df, x="Tanggal", y="Prediksi Jumlah", title="Bar Chart Prediksi")
            fig_bar.update_layout(
                hovermode="x unified",
                dragmode="select",
                selectdirection="h"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Area Chart
            st.subheader("🌊 Area Chart")
            fig_area = px.area(forecast_df, x="Tanggal", y="Prediksi Jumlah", title="Area Chart Prediksi")
            fig_area.update_layout(
                hovermode="x unified",
                dragmode="select",
                selectdirection="h"
            )
            st.plotly_chart(fig_area, use_container_width=True)

            # 📋 TABEL HASIL PREDIKSI (dengan kolom No)
            st.subheader("📋 Tabel Hasil Prediksi")
            tabel_prediksi = forecast_df.copy().reset_index(drop=True)
            tabel_prediksi.insert(0, "No", range(1, len(tabel_prediksi) + 1))
            st.dataframe(tabel_prediksi.style.format({"Prediksi Jumlah": "{:,.0f}"}))
            
            # Distribusi Prediksi Bulanan
            st.subheader("Status Prediksi Bulanan")
            lead_status_fig = px.bar(
                forecast_df,
                x=forecast_df["Tanggal"].dt.strftime("%b %Y"),
                y="Prediksi Jumlah",
                title="Distribusi Prediksi per Bulan"
            )
            lead_status_fig.update_layout(
                hovermode="x unified",
                dragmode="select",
                selectdirection="h"
            )
            st.plotly_chart(lead_status_fig, use_container_width=True)

            # Trend Prediksi Barang
            st.subheader("Trend Prediksi Barang per Bulan")
            fig_campaign = px.line(forecast_df, x="Tanggal", y="Prediksi Jumlah", title="Trend Prediksi per Bulan")
            fig_campaign.update_layout(
                hovermode="x unified",
                dragmode="select",
                selectdirection="h"
            )
            st.plotly_chart(fig_campaign, use_container_width=True)

            # Pendapatan Tahunan (Akumulasi)
            forecast_df["Tahun"] = forecast_df["Tanggal"].dt.year
            yearly_income = forecast_df.groupby("Tahun")["Prediksi Jumlah"].sum().reset_index()

            st.subheader("Pendapatan Prediksi Tahunan")
            fig_year = px.line(yearly_income, x="Tahun", y="Prediksi Jumlah", markers=True, title="Pendapatan Prediksi Tahunan")
            fig_year.update_layout(
                hovermode="x unified",
                dragmode="select",
                selectdirection="h"
            )
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
            fig_kuartal.update_layout(hovermode="closest")
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
                fig_proporsi.update_layout(hovermode="closest")
                st.plotly_chart(fig_proporsi, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
