import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import time
import sys

# ==========================================
# 1. CORE LOGIC: SIGNAL PROCESSING & ML
# ==========================================
class PotholeDetector:
    """
    Implements the detection logic described in Phase II Report[cite: 350].
    Uses vibration analysis (Z-axis accelerometer) to identify potholes[cite: 350].
    """
    def __init__(self):
        # Thresholds based on typical accelerometer gravity (9.8 m/s^2)
        self.POTHOLE_THRESHOLD = 3.5  # Variance threshold for "jolt"
        self.GRAVITY = 9.8

    def extract_features(self, window):
        """
        Extracts features (Mean, Std, RMS) from a sensor window as detailed in 
        Technical Explanation[cite: 77].
        """
        # Remove gravity effect roughly by centering
        z_dynamic = window['az'] - window['az'].mean()
        
        features = {
            'std_dev': np.std(z_dynamic),
            'peak_to_peak': np.ptp(z_dynamic),
            'rms': np.sqrt(np.mean(z_dynamic**2)),
            'max_jolt': np.max(np.abs(z_dynamic))
        }
        return features

    def detect(self, df_window):
        """
        Returns probability of pothole (0.0 to 1.0) based on decision logic[cite: 93].
        """
        feats = self.extract_features(df_window)
        
        # Simple heuristic: High Standard Deviation = Rough Road/Pothole
        severity = 0.0
        if feats['std_dev'] > self.POTHOLE_THRESHOLD:
            # Scale severity between 0.7 and 1.0 based on intensity
            severity = min(1.0, 0.7 + (feats['std_dev'] - self.POTHOLE_THRESHOLD) / 10)
        else:
            # Smooth road
            severity = 0.1
            
        return severity, feats

# ==========================================
# 2. DATA SIMULATION (MOCK SENSORS)
# ==========================================
def generate_trip_data(num_points=200):
    """
    Simulates GPS and Accelerometer data[cite: 417].
    Generates a path starting from a dummy location (Bangalore).
    """
    # Start at Bangalore coordinates
    lat, lon = 12.9716, 77.5946
    
    data = []
    
    for i in range(num_points):
        # Move slightly to simulate driving
        lat += np.random.normal(0, 0.0001)
        lon += np.random.normal(0, 0.0001)
        
        # Simulate Accelerometer Z-axis (up/down)
        az = np.random.normal(9.8, 0.5)
        
        # Inject random Potholes (Sudden spikes)
        if np.random.rand() < 0.1:  # 10% chance of pothole
            az += np.random.choice([-1, 1]) * np.random.uniform(2.5, 4.0)
            
        data.append({
            'timestamp': time.time() + i,
            'lat': lat,
            'lon': lon,
            'az': az
        })
        
    return pd.DataFrame(data)

# ==========================================
# 3. DASHBOARD & MAPPING UI
# ==========================================
def main():
    st.set_page_config(page_title="Pothole Mapping System", layout="wide")
    
    # Initialize session state for data storage
    if 'data' not in st.session_state:
        st.session_state['data'] = generate_trip_data(200)

    # Header based on Report Title
    st.title("🚦 Pothole Mapping System - Dashboard")
    st.markdown("""
    **Phase-II Prototype Implementation** *Integrates Sensor Analysis, Detection Logic, and Interactive Mapping.* 
    """)
    
    # Sidebar: Controls
    st.sidebar.header("Control Panel")
    data_source = st.sidebar.radio("Data Source", ["Simulate Live Trip", "Upload CSV"])
    
    df = pd.DataFrame()

    # --- Data Acquisition Logic ---
    if data_source == "Simulate Live Trip":
        df = st.session_state['data']
        if st.sidebar.button("Regenerate Trip Data"):
             st.session_state['data'] = generate_trip_data(200)
             # Rerun the script to apply changes
             st.rerun() 
        
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Sensor Log (CSV)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a file to proceed with analysis.")
            return # Exit if no file is uploaded

    # --- Data Processing and Visualization Block ---
    if not df.empty:
        detector = PotholeDetector()
        results = []
        
        # Sliding window analysis (Simulating the 1s window buffer [cite: 66])
        window_size = 10 
        for i in range(0, len(df) - window_size, window_size):
            window = df.iloc[i:i+window_size]
            severity, feats = detector.detect(window)
            
            # Use the location of the middle of the window
            center_idx = i + window_size // 2
            results.append({
                'lat': df.iloc[center_idx]['lat'],
                'lon': df.iloc[center_idx]['lon'],
                'severity': severity,
                'std_dev': feats['std_dev']
            })
            
        results_df = pd.DataFrame(results)
        
        # 3. Visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Interactive Pothole Map")
            
            # Center the map on the data's mean location
            center_lat = df['lat'].mean() if not df.empty else 12.9716
            center_lon = df['lon'].mean() if not df.empty else 77.5946
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
            
            # Prepare data for the heatmap (lat, lon, weight/severity)
            heat_data = results_df[['lat', 'lon', 'severity']].values.tolist()
            
            # Apply color mapping: High severity (Red), Low severity (Green)
            HeatMap(
                heat_data, 
                radius=15, 
                max_zoom=13, 
                # This ensures the gradient goes from green (0) to red (1)
                gradient={0.1: 'green', 0.5: 'yellow', 0.7: 'orange', 1.0: 'red'}
            ).add_to(m)
            
            # Add specific markers for confirmed potholes
            potholes = results_df[results_df['severity'] > 0.6]
            for _, row in potholes.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5,
                    color="#FF4500", # Red-Orange
                    fill=True,
                    popup=f"Pothole Detected! Severity: {row['severity']:.2f}"
                ).add_to(m)
                
            st_folium(m, width=800, height=500)
            
            st.markdown("---")
            st.markdown("Heatmap color key: **Green** (Smooth Road) → **Yellow/Orange** (Moderate Roughness) → **Red** (Confirmed Pothole)")
            


        with col2:
            st.subheader("Trip Statistics")
            total_potholes = len(potholes)
            avg_severity = results_df['severity'].mean()
            
            st.metric("Potholes Detected", total_potholes, delta_color="inverse")
            st.metric("Average Road Roughness", f"{avg_severity:.2}")
            
            st.subheader("Sensor Analysis (Z-Axis)")
            st.line_chart(df['az'].head(100)) # Show raw sensor data for visualization
            st.caption("Raw accelerometer data showing vertical jolts (spikes indicate potholes).")

    else:
        # This message should ideally never be reached on the first run now.
        st.error("Error: Simulation data could not be generated. Check environment dependencies.")


if __name__ == "__main__":
    main()