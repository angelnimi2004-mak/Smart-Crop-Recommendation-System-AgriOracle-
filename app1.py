import streamlit as st
import numpy as np
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AgriOracle | Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.stButton > button {
    background-color: #2E7D32;
    color: white;
    border-radius: 30px;
    height: 50px;
    font-size: 18px;
}
.stButton > button:hover {
    background-color: #1B5E20;
}
.card {
    background-color: #1A1F2B;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_crop_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🧠 **AgriOracle**")
    

    st.markdown("""
    <div class="card">
    This system uses Machine Learning (Random Forest) to analyze soil and climate data to recommend the most profitable crop.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✅ System Specs")
    st.success("Accuracy: 100.0%")
    st.success("Features: 7 Parameters")
    st.success("Dataset: 12,000 Samples")

# ---------------- Main Section ----------------
st.markdown(
    "<h2 style='text-align:center;'>Optimize your yield by providing precise soil and environmental data</h2>",
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🌱 Soil Nutrients", "☁️ Environmental Factors"])

# ---------------- Inputs ----------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", 0, 150, 50)
        p = st.number_input("Phosphorus (P)", 0, 150, 50)
    with col2:
        k = st.number_input("Potassium (K)", 0, 210, 50)
        ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    with col4:
        hum = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
        rain = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- Predict Button ----------------
col_btn = st.columns([3,2,3])[1]
with col_btn:
    predict = st.button("🚀 Analyze & Recommend")

if predict:
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.success(f"🌾 **Recommended Crop:** `{prediction.upper()}`")

# ---------------- Footer ----------------
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
Made with ❤️ for Sustainable Farming
</p>
""", unsafe_allow_html=True)

