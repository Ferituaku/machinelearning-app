import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Global Economy Cluster Prediction",
    page_icon="🌍",
    layout="wide"
)

# Define cluster information at the start
CLUSTER_INFO = {
    0: {
        "name": "Developing Economy",
        "description": "Medium Performance",
        "color": "#FFA07A"
    },
    1: {
        "name": "Emerging Economy",
        "description": "Lower Performance",
        "color": "#98FB98"
    },
    2: {
        "name": "Advanced Economy",
        "description": "Higher Performance",
        "color": "#87CEEB"
    }
}

# Title and description
st.title("🌍 Global Economy Cluster Prediction")
st.markdown("""
This application predicts the economic cluster of a country based on key economic indicators.
Please input the values for each indicator using the sliders below.
""")

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('kmeans_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error: Model files not found. Please ensure kmeans_model.pkl and scaler.pkl are in the same directory.")
        st.stop()

model, scaler = load_model()

# Create input form
st.subheader("Input Country Data")

# Input for country name
country_name = st.text_input("Country Name", "")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    social_capital = st.slider(
        "Social Capital Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="Measure of social networks and community engagement"
    )
    
    governance = st.slider(
        "Governance Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="Measure of government effectiveness and institutional quality"
    )

with col2:
    economic_quality = st.slider(
        "Economic Quality Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="Measure of economic performance and stability"
    )
    
    living_conditions = st.slider(
        "Living Conditions Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="Measure of quality of life and standard of living"
    )

# Create prediction button
if st.button("Predict Cluster"):
    if country_name.strip() == "":
        st.warning("Please enter a country name.")
    else:
        try:
            # Prepare input data
            input_data = np.array([[
                social_capital,
                governance,
                economic_quality,
                living_conditions
            ]])
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            cluster = model.predict(input_scaled)[0]
            
            # Get cluster information
            cluster_data = CLUSTER_INFO.get(cluster, {
                "name": "Unknown",
                "description": f"Cluster {cluster}",
                "color": "#CCCCCC"
            })
            
            # Show prediction with custom styling
            st.markdown("---")
            st.subheader("Prediction Results")
            
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {cluster_data['color']};">
                    <h3 style="color: black;">Cluster Prediction for {country_name}</h3>
                    <p style="color: black; font-size: 18px;">
                        {cluster_data['name']}: {cluster_data['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Show input values summary
            st.markdown("### Input Summary")
            summary_df = pd.DataFrame({
                'Indicator': ['Social Capital', 'Governance', 'Economic Quality', 'Living Conditions'],
                'Score': [social_capital, governance, economic_quality, living_conditions]
            })
            st.table(summary_df)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Add footer with information
st.markdown("---")
st.markdown("""
### About
This model uses K-means clustering with 3 clusters to categorize countries based on their economic indicators.
The prediction is based on four key features identified through PCA analysis:
- Social Capital
- Governance
- Economic Quality
- Living Conditions

*Note: This is a simplified model for demonstration purposes. Results should be interpreted with appropriate context.*
""")
