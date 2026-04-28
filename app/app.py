import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. THE SMART ENGINEERING FUNCTION ---
# This ensures custom features are calculated exactly as they were during training.
def smart_engineering(X):
    X_df = pd.DataFrame(X).copy()

    # Calculate engineered features if source dimensions exist
    if all(k in X_df.columns for k in ['x', 'y', 'z']):
        X_df['volume'] = X_df['x'] * X_df['y'] * X_df['z']
        X_df['l_w_ratio'] = X_df['x'] / (X_df['y'] + 1e-6)

    # List of columns the model expects in the final stage
    expected_cols = ['carat', 'depth', 'table', 'volume', 'l_w_ratio', 'cut', 'color', 'clarity']

    # Return only the columns that exist in the current dataframe to prevent KeyErrors
    existing_cols = [c for c in expected_cols if c in X_df.columns]
    return X_df[existing_cols]


# --- 2. MODEL LOADING ---
st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="wide")

@st.cache_resource
def load_final_model():
    # Ensure 'diamond_final_pro_model.pkl' is in the same folder as this script
    return joblib.load('app/model.pkl')

try:
    model = load_final_model()
except Exception as e:
    st.error(f"Critical Error: Could not load the model file. {e}")
    st.stop()

# --- 3. USER INTERFACE ---
st.title("💎 Professional Diamond Valuation Tool")
st.markdown("Enter the diamond specifications below for a real-time market price estimation.")

# Layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Weight & Cut")
    # Removed max constraints to allow entry of any size, kept min at 0.0
    carat = st.number_input("Carat Weight", min_value=0.0, value=1.0, step=0.01)
    cut = st.selectbox("Cut Quality", ["Ideal", "Premium", "Very Good", "Good", "Fair"])

with col2:
    st.subheader("Quality Grade")
    color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity Grade", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

with col3:
    st.subheader("Proportions")
    # Broadened ranges for Depth and Table
    depth = st.number_input("Depth %", min_value=0.0, max_value=100.0, value=61.0, step=0.1)
    table = st.number_input("Table %", min_value=0.0, max_value=100.0, value=57.0, step=0.1)

st.markdown("---")
st.subheader("Physical Dimensions (mm)")
d_col1, d_col2, d_col3 = st.columns(3)
with d_col1:
    x = st.number_input("Length (x)", min_value=0.0, value=6.4)
with d_col2:
    y = st.number_input("Width (y)", min_value=0.0, value=6.4)
with d_col3:
    z = st.number_input("Depth (z)", min_value=0.0, value=4.0)

# --- 4. PREDICTION LOGIC ---
if st.button("Generate Valuation Report", use_container_width=True):
    # Constructing the DataFrame with keys that match your training data
    input_df = pd.DataFrame([{
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }])

    try:
        # Run inference through the pipeline
        prediction = model.predict(input_df)

        # UI Results
        st.balloons()
        st.markdown("---")
        st.success(f"### 🎯 Estimated Market Price: **${prediction[0]:,.2f}**")

        # Technical Breakdown
        with st.expander("📊 View Model Analysis"):
            vol = x * y * z
            st.write(f"**Calculated Volume:** {vol:.2f} mm³")
            st.write(f"**L/W Ratio:** {x / (y + 1e-6):.2f}")
            st.write("**Model Type:** XGBoost Regressor")
            st.write("**Data Source:** Scraped Market Data (54k records)")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure your model was saved with the same column names used here.")

# --- FOOTER ---
st.markdown("---")
st.caption("Project 2: Diamond Price Analysis | Built with Streamlit & XGBoost")
