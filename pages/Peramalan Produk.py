import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

        df_agg = df_cleaned.groupby('Nama Barang', as_index=False).agg({'Jumlah': 'sum'})
        top_items = df_agg.nlargest(5, 'Jumlah').reset_index(drop=True)

        st.write("5 Barang dengan Jumlah Unit Terbanyak:")
        top_display = top_items[['Nama Barang', 'Jumlah']].copy()
        top_display.insert(0, 'No', range(1, len(top_display) + 1))
        st.dataframe(top_display)

        fig_top = px.bar(top_items,
                         x='Nama Barang',
                         y='Jumlah',
                         text='Jumlah',
                         title="Top 5 Barang dengan Jumlah Unit Terbanyak",
                         labels={'Jumlah': 'Jumlah Unit'},
                         template='plotly_white')
        fig_top.update_traces(marker_color='indigo', hovertemplate='%{x}<br>Jumlah: %{y}')
        fig_top.update_layout(
            xaxis_tickangle=-45,
            height=500,
            margin=dict(l=40, r=40, t=60, b=120),
            hovermode="x unified"
        )
        st.plotly_chart(fig_top, use_container_width=True)

        top_products = df_cleaned[df_cleaned['Nama Barang'].isin(top_items['Nama Barang'])]
        top_products['Bulan'] = top_products['Tanggal'].dt.to_period('M')
        top_products_grouped = top_products.groupby(['Bulan', 'Nama Barang'])['Jumlah'].sum().reset_index()
        top_products_grouped['Bulan'] = pd.to_datetime(top_products_grouped['Bulan'].astype(str))

        st.markdown("### Tren Permintaan 5 Produk Teratas (Agregasi Bulanan)")
        fig_trend = px.line(top_products_grouped,
                            x='Bulan',
                            y='Jumlah',
                            color='Nama Barang',
                            markers=True,
                            template='plotly_white')
        fig_trend.update_traces(hovertemplate='%{x|%B %Y}<br>%{y}')
        fig_trend.update_layout(
            height=500,
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=60),
            legend=dict(title="Nama Barang", orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            hovermode="x unified",
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

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

        st.subheader("Pilih Metode Peramalan")
        mode = st.radio("Metode:", ["Top 5 Produk Teratas", "Pilih Produk Sendiri"])

        selected_products = []
        if mode == "Top 5 Produk Teratas":
            selected_products = [p for p in top_items['Nama Barang'] if p in adf_pass_products]

            if not selected_products:
                st.warning("Tidak ada dari Top 5 produk yang lolos uji ADF. Silakan pilih produk sendiri.")

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

        if selected_products and st.button("Lakukan Peramalan"):
            for product in selected_products:
                st.write(f"\n### Peramalan untuk {product}")
                product_data_resampled = resampled_products[product]

                max_lags = min(40, len(product_data_resampled.dropna()) // 2)

                # Plot ACF & PACF Interaktif
                st.subheader("Plot ACF & PACF")

                acf_values = sm.tsa.stattools.acf(product_data_resampled.dropna(), nlags=max_lags)
                pacf_values = sm.tsa.stattools.pacf(product_data_resampled.dropna(), nlags=max_lags)

                acf_fig = px.bar(
                    x=list(range(len(acf_values))),
                    y=acf_values,
                    labels={'x': 'Lag', 'y': 'ACF'},
                    title=f"Autocorrelation (ACF) untuk {product}"
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
                    title=f"Partial Autocorrelation (PACF) untuk {product}"
                )
                pacf_fig.update_layout(
                    hovermode="x unified",
                    dragmode="select",
                    selectdirection="h",
                    showlegend=False
                )

                # Menampilkan Plot ACF dan PACF berdampingan
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(acf_fig, use_container_width=True)
                with col2:
                    st.plotly_chart(pacf_fig, use_container_width=True)

                model = SARIMAX(product_data_resampled,
                                order=(1, 0, 1),
                                seasonal_order=(1, 0, 1, 52),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit()

                forecast_object = results.get_forecast(steps=52)
                forecast_values = forecast_object.predicted_mean
                forecast_ci = forecast_object.conf_int()

                if not isinstance(forecast_values.index, pd.DatetimeIndex):
                    last_date = product_data_resampled.index[-1]
                    forecast_values.index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=52, freq='W')
                    forecast_ci.index = forecast_values.index

                forecast_values = forecast_values.clip(lower=0)
                forecast_ci = forecast_ci.clip(lower=0)

                fig_pred = px.line(title=f"Prediksi SARIMA (12 Bulan ke Depan) untuk {product}",
                                   template='plotly_white')
                fig_pred.add_scatter(x=product_data_resampled.index, y=product_data_resampled.values,
                                     mode='lines+markers', name='Data Aktual')
                fig_pred.add_scatter(x=forecast_values.index, y=forecast_values.values,
                                     mode='lines+markers', name='Prediksi SARIMA')
                fig_pred.add_scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0],
                                     mode='lines', line=dict(width=0), showlegend=False)
                fig_pred.add_scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1],
                                     mode='lines', fill='tonexty',
                                     fillcolor='rgba(173, 216, 230, 0.4)',
                                     line=dict(width=0), name='Confidence Interval')
                fig_pred.update_layout(hovermode="x unified")
                st.plotly_chart(fig_pred, use_container_width=True)

                forecast_df = pd.DataFrame({
                    'Minggu': forecast_values.index,
                    'Prediksi Jumlah': np.round(forecast_values.values).astype(int)
                })

                fig_bar = px.bar(forecast_df,
                                 x='Minggu',
                                 y='Prediksi Jumlah',
                                 text='Prediksi Jumlah',
                                 title=f"Bar Chart Prediksi Jumlah per Minggu untuk {product}",
                                 template='plotly_white')
                fig_bar.update_traces(marker_color='dodgerblue', hovertemplate='%{x|%Y-%m-%d}<br>Prediksi: %{y}')
                fig_bar.update_layout(xaxis_tickformat='%Y-%m-%d', xaxis_tickangle=-45, hovermode="x unified")
                st.plotly_chart(fig_bar, use_container_width=True)

                forecast_df_display = forecast_df.copy()
                forecast_df_display['Minggu'] = forecast_df_display['Minggu'].dt.strftime('%Y-%m-%d')
                forecast_df_display.insert(0, 'No', range(1, len(forecast_df_display) + 1))

                st.write(f"Tabel Hasil Prediksi SARIMA untuk {product}:")
                st.dataframe(forecast_df_display)
