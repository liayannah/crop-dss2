import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px

st.title("🌾 Crop Management & 🐛 Pest Control DSS")
st.caption("Upload your own data to train models and get predictions with insights.")

st.sidebar.header("📤 Upload CSV Data")
crop_file = st.sidebar.file_uploader("Upload Crop Data CSV", type="csv")
pest_file = st.sidebar.file_uploader("Upload Pest Data CSV", type="csv")

# Function to train regression
def train_model(df, target_col, feature_cols):
    X = sm.add_constant(df[feature_cols])
    y = df[target_col]
    model = sm.OLS(y, X).fit()
    return model

if crop_file:
    crop_data = pd.read_csv(crop_file)
    st.subheader("📊 Crop Data Preview")
    st.dataframe(crop_data)

    # Train model
    crop_features = ['Fertilizer', 'Irrigation', 'Labor', 'SoilQuality', 'WeatherIndex']
    model_yield = train_model(crop_data, 'Yield', crop_features)

    # Input for yield prediction
    st.markdown("### 🔍 Yield Prediction Input")
    user_input = {col: st.number_input(col, value=float(crop_data[col].mean())) for col in crop_features}
    input_yield = pd.DataFrame([user_input])
    input_yield = input_yield.reindex(columns=model_yield.model.exog_names, fill_value=1)
    yield_prediction = model_yield.predict(input_yield)[0]
    st.success(f"📈 Predicted Crop Yield: {yield_prediction:.2f} kg/ha")

    # Yield vs Fertilizer Chart
    st.markdown("### 📈 Crop Yield vs Fertilizer")
    fig1 = px.scatter(crop_data, x="Fertilizer", y="Yield", trendline="ols", color_discrete_sequence=["green"])
    st.plotly_chart(fig1, use_container_width=True)

if pest_file:
    pest_data = pd.read_csv(pest_file)
    st.subheader("🦠 Pest Data Preview")
    st.dataframe(pest_data)

    pest_features = ['Temperature', 'Humidity', 'CropStage', 'Pesticide', 'TimeSinceSpray']
    model_pest = train_model(pest_data, 'PestIncidence', pest_features)

    # Input for pest prediction
    st.markdown("### 🔍 Pest Incidence Prediction Input")
    pest_input = {col: st.number_input(col, value=float(pest_data[col].mean())) for col in pest_features}
    input_pest = pd.DataFrame([pest_input])
    input_pest = input_pest.reindex(columns=model_pest.model.exog_names, fill_value=1)
    pest_prediction = model_pest.predict(input_pest)[0]
    st.warning(f"🐛 Predicted Pest Incidence: {pest_prediction:.2f} pests/m²")

    # Pest vs Humidity Chart
    st.markdown("### 📉 Pest Incidence vs Humidity")
    fig2 = px.scatter(pest_data, x="Humidity", y="PestIncidence", trendline="ols", color_discrete_sequence=["red"])
    st.plotly_chart(fig2, use_container_width=True)

st.sidebar.markdown("ℹ️ Use the CSV templates provided.")
