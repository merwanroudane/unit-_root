import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, range_unit_root_test
from arch.unitroot import PhillipsPerron
import statsmodels.api as sm


def main():
    st.title("Time Series Unit Root Tests - Dr. Merwan Roudane")  # Added name
    st.write("""
    This app performs various unit root tests on your time series data.
    Upload an Excel file, select a variable, and choose which tests to run.
    """)

    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Read the data
        df = pd.read_excel(uploaded_file)

        # Display the dataframe
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        # Select columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            st.error("No numeric columns found in the uploaded file.")
            return

        selected_column = st.selectbox("Select a variable for unit root testing:", numeric_columns)

        # Get the data
        data = df[selected_column].dropna()

        # Plot the time series
        st.write("### Time Series Plot:")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.values)
        ax.set_title(f'Time Series: {selected_column}')
        st.pyplot(fig)

        # Options for tests
        st.write("### Select Unit Root Tests to Perform:")
        run_adf = st.checkbox("Augmented Dickey-Fuller (ADF) Test", value=True)
        run_kpss = st.checkbox("KPSS Test", value=True)
        run_pp = st.checkbox("Phillips-Perron (PP) Test", value=True)
        run_rur = st.checkbox("Range Unit Root Test", value=True)

        if st.button("Run Tests"):
            st.write("## Test Results")

            # ADF Test
            if run_adf:
                st.write("### Augmented Dickey-Fuller Test")
                st.write("""
                **Null Hypothesis**: The series has a unit root (non-stationary)  
                **Alternative Hypothesis**: The series has no unit root (stationary)
                """)

                result = adfuller(data.values)

                adf_output = pd.DataFrame({
                    'Statistic': [result[0]],
                    'p-value': [result[1]],
                    'Lags Used': [result[2]],
                    'Number of Observations': [result[3]]
                })

                for key, value in result[4].items():
                    adf_output[f'Critical Value ({key})'] = value

                st.dataframe(adf_output)

                if result[1] <= 0.05:
                    st.success("The series is stationary (reject the null hypothesis).")
                else:
                    st.warning("The series is non-stationary (fail to reject the null hypothesis).")

            # KPSS Test
            if run_kpss:
                st.write("### KPSS Test")
                st.write("""
                **Null Hypothesis**: The series is stationary  
                **Alternative Hypothesis**: The series has a unit root (non-stationary)
                """)

                try:
                    result = kpss(data.values, regression='c')

                    kpss_output = pd.DataFrame({
                        'Statistic': [result[0]],
                        'p-value': [result[1]],
                        'Lags Used': [result[2]],
                    })

                    for key, value in result[3].items():
                        kpss_output[f'Critical Value ({key})'] = value

                    st.dataframe(kpss_output)

                    if result[1] <= 0.05:
                        st.warning("The series is non-stationary (reject the null hypothesis).")
                    else:
                        st.success("The series is stationary (fail to reject the null hypothesis).")
                except Exception as e:
                    st.error(f"Error in KPSS test: {str(e)}")

            # Phillips-Perron Test (using arch library)
            if run_pp:
                st.write("### Phillips-Perron Test")
                st.write("""
                **Null Hypothesis**: The series has a unit root (non-stationary)  
                **Alternative Hypothesis**: The series has no unit root (stationary)
                """)

                try:
                    # Using the correct Phillips-Perron test from arch library
                    pp = PhillipsPerron(data.values)
                    pp_result = pp.summary()

                    # Extract key info from the result
                    pp_output = pd.DataFrame({
                        'Statistic': [pp.stat],
                        'p-value': [pp.pvalue],
                        'Lags Used': [pp.lags],
                    })

                    # Add critical values
                    critical_values = pp.critical_values
                    for key in critical_values.keys():
                        pp_output[f'Critical Value ({key}%)'] = critical_values[key]

                    st.dataframe(pp_output)

                    if pp.pvalue <= 0.05:
                        st.success("The series is stationary (reject the null hypothesis).")
                    else:
                        st.warning("The series is non-stationary (fail to reject the null hypothesis).")

                    # Detailed PP test results
                    st.text("Detailed Phillips-Perron Test Results:")
                    st.text(str(pp_result))

                except Exception as e:
                    st.error(f"Error in Phillips-Perron test: {str(e)}")
                    st.info("Note: This test requires the 'arch' library. Install it using pip install arch")

            # Range Unit Root Test
            if run_rur:
                st.write("### Range Unit Root Test")
                st.write("""
                **Null Hypothesis**: The series has a unit root (non-stationary)  
                **Alternative Hypothesis**: The series has no unit root (stationary)
                """)  # Corrected Hypotheses

                try:
                    result = range_unit_root_test(data.values, store=True)

                    rur_output = pd.DataFrame({
                        'Statistic': [result[0]],
                        'p-value': [result[1]],
                    })

                    # Add critical values
                    for key, value in result[2].items():
                        rur_output[f'Critical Value ({key})'] = value

                    st.dataframe(rur_output)

                    if result[1] <= 0.05:
                        st.success("The series is stationary (reject the null hypothesis).")  # Corrected Interpretation
                    else:
                        st.warning(
                            "The series is non-stationary (fail to reject the null hypothesis).")  # Corrected Interpretation

                except Exception as e:
                    st.error(f"Error in Range Unit Root test: {str(e)}")


if __name__ == "__main__":
    main()