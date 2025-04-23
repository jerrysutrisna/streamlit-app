import streamlit as st
import pickle
import pandas as pd

# Load model SARIMA
with open('modelsarima.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Multipage App",
    page_icon="ok"
)

# File uploader
uploaded_file = st.file_uploader("Upload Excel file to include in prediction", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    input_data = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(input_data)

# Input interface
st.title("Forecasting with SARIMA")
input_periods = st.number_input("Forecast Periods:", min_value=1, max_value=100, value=10)

        # Run prediction
if st.button("Predict"):
            try:
                # Generate forecast
                forecast = model.forecast(steps=input_periods)
                forecast_index = pd.date_range(start=input_data.index[-1], periods=input_periods + 1, freq='M')[1:]
                forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})

                st.write("Forecast Results:")
                st.line_chart(forecast_df.set_index('Date'))
                
                # Save forecast to Excel
                output_filename = 'forecast_results.xlsx'
                forecast_df.to_excel(output_filename, index=False)
                
                with open(output_filename, 'rb') as output_file:
                    st.download_button(
                        label="Download Forecast Results",
                        data=output_file,
                        file_name=output_filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

            except Exception as e:
                st.error(f"Error during prediction: {e}")
else:
        st.error("Uploaded file must contain a 'Date' column and a 'Value' column.")
