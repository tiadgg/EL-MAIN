import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import time
import random
from geopy.distance import great_circle

# ==========================================
# 1. CORE LOGIC: SIGNAL PROCESSING & DETECTION
# (Based on App Explanation and Phase II Reports)
# ==========================================
class PotholeDetector:
    """
    Simulates the ML-based detection using vibration data (Z-axis accelerometer) 
    as described in the project documentation[cite: 43, 417, 139].
    """
    def __init__(self):
        # Thresholds tuned for detection severity based on vibration analysis [cite: 43, 61, 181]
        self.LOW_THRESHOLD = 2.0   # Marks as Yellow/Green
        self.HIGH_THRESHOLD = 3.5  # Marks as Red

    def extract_features(self, window):
        """
        Extracts features (Standard Deviation of acceleration) from a sensor window[cite: 164].
        """
        # Focus on the dynamic part of Z-axis acceleration
        z_dynamic = window['az'] - window['az'].mean()
        features = {
            'std_dev': np.std(z_dynamic)
        }
        return features

    def detect(self, df_window):
        """
        Calculates severity based on vibration magnitude.
        Returns severity (0.0 to 1.0) and the corresponding color key (Red, Yellow, Green).
        """
        feats = self.extract_features(df_window)
        std_dev = feats['std_dev']

        if std_dev >= self.HIGH_THRESHOLD:
            # High Jolt: Red (Confirmed Pothole/Obstacle)
            severity = 1.0
            color = 'red'
        elif std_dev >= self.LOW_THRESHOLD:
            # Moderate Jolt: Yellow (Rough Patch/Minor Bumps)
            severity = 0.6
            color = 'yellow'
        else:
            # Low Jolt: Green (Smooth Road)
            severity = 0.2
            color = 'green'
            
        return severity, color, feats

# ==========================================
# 2. DATA SIMULATION FOR SPECIFIC BANGALORE ROADS
# ==========================================

# Define Waypoints for MG Road -> Electronic City -> Mysore Road (Approximate Route)
BANGALORE_WAYPOINTS = [
    # MG Road (Start)
    (12.9757, 77.6062),
    (12.9734, 77.6042),
    # Moving towards Electronic City
    (12.9348, 77.6253),
    # Electronic City Flyover/Road
    (12.8529, 77.6603),
    (12.8468, 77.6534),
    # Towards Mysore Road
    (12.9248, 77.5855),
    # Mysore Road (End near NICE Road junction)
    (12.9366, 77.5345),
    (12.9405, 77.5256)
]

def generate_trip_data_bangalore(waypoints, pothole_density=0.15):
    """
    Simulates GPS and Accelerometer data along the specified waypoints[cite: 36].
    """
    data = []
    
    for i in range(len(waypoints) - 1):
        start_lat, start_lon = waypoints[i]
        end_lat, end_lon = waypoints[i+1]
        
        # Calculate distance and steps between waypoints
        distance = great_circle((start_lat, start_lon), (end_lat, end_lon)).kilometers
        num_steps = max(20, int(distance * 500)) # More steps for longer segments

        lat_step = (end_lat - start_lat) / num_steps
        lon_step = (end_lon - start_lon) / num_steps
        
        # Interpolate between points
        for j in range(num_steps):
            lat = start_lat + lat_step * j + np.random.normal(0, 0.00001)
            lon = start_lon + lon_step * j + np.random.normal(0, 0.00001)
            
            # --- Accelerometer Simulation ---
            az = np.random.normal(9.8, 0.4) # Base smooth road noise
            
            # Inject varying jolts based on simulated road condition
            if random.random() < pothole_density:
                if random.random() < 0.3: 
                    # Simulates a severe pothole (Red)
                    az += np.random.choice([-1, 1]) * np.random.uniform(3.0, 5.0) 
                else:
                    # Simulates a minor bump (Yellow)
                    az += np.random.choice([-1, 1]) * np.random.uniform(1.5, 2.5) 
            
            data.append({
                'lat': lat,
                'lon': lon,
                'az': az
            })
            
    return pd.DataFrame(data)

# ==========================================
# 3. STREAMLIT DASHBOARD & MAPPING UI
# (Government Dashboard Access for Proactive Repair Planning) [cite: 428, 418, 26]
# ==========================================
def main():
    st.set_page_config(page_title="Bangalore Pothole Heatmap", layout="wide")
    
    # Initialize session state with data if not present
    if 'bangalore_data' not in st.session_state:
        st.session_state['bangalore_data'] = generate_trip_data_bangalore(BANGALORE_WAYPOINTS)

    st.title("🗺️ Real-time Pothole Mapping System - Bangalore Roads Dashboard")
    st.markdown("---")
    
    df = st.session_state['bangalore_data']
    
    if st.sidebar.button("Regenerate Trip Data"):
        st.session_state['bangalore_data'] = generate_trip_data_bangalore(BANGALORE_WAYPOINTS)
        st.rerun() 
    
    st.sidebar.caption("Route: MG Road -> Electronic City Flyover -> Mysore Road")

    # --- Data Processing ---
    detector = PotholeDetector()
    results = []
    window_size = 10 
    
    with st.spinner("Processing sensor data and running ML inference..."):
        for i in range(0, len(df) - window_size, window_size):
            window = df.iloc[i:i+window_size]
            severity, color, feats = detector.detect(window)
            
            center_idx = i + window_size // 2
            results.append({
                'lat': df.iloc[center_idx]['lat'],
                'lon': df.iloc[center_idx]['lon'],
                'severity': severity,
                'color': color,
                'std_dev': feats['std_dev']
            })
            
    results_df = pd.DataFrame(results)
    
    # --- Visualization ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Interactive Road Condition Heatmap (Leaflet)")
        
        # Center the map near Central Bangalore
        center_lat, center_lon = 12.95, 77.58 
        
        # FIX: Using "OpenStreetMap" as the default tile set resolves the Folium attribution error.
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
        
        # Prepare data for the heatmap: only severe and moderate bumps contribute heavily
        heat_data_points = results_df[results_df['severity'] > 0.4][['lat', 'lon', 'severity']].values.tolist()
        
        # Heatmap layer for general road roughness/damage severity [cite: 427, 423]
        HeatMap(
            heat_data_points, 
            radius=20, 
            max_zoom=13, 
            # Custom gradient reflecting the severity: Green (smooth) -> Yellow (moderate) -> Red (severe)
            gradient={0.0: 'green', 0.5: 'yellow', 0.8: 'red'}
        ).add_to(m)
        
        
        # Add distinct Circle Markers for high severity points (Red and Yellow)
        potholes = results_df[results_df['color'] == 'red']
        rough_patches = results_df[results_df['color'] == 'yellow']

        for _, row in rough_patches.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color="yellow",
                fill=True,
                fillOpacity=0.7
            ).add_to(m)

        for _, row in potholes.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6,
                color="#FF0000", # Pure Red
                fill=True,
                fillOpacity=1.0,
                popup=f"Severe Pothole/Obstacle! (StdDev: {row['std_dev']:.2f})"
            ).add_to(m)

        st_folium(m, width=900, height=550)
        
        st.markdown("""
        **Color Legend (Road Obstacle/Pothole Density):**
        * 🔴 **Red:** Too many potholes and obstacles (High Severity)
        * 🟡 **Yellow:** Little lesser (Moderate Roughness)
        * 🟢 **Green:** Even lesser (Smooth Road)
        """)

    with col2:
        st.subheader("Trip Statistics (for Repair Planning) [cite: 418]")
        total_points = len(results_df)
        red_count = len(potholes)
        yellow_count = len(rough_patches)
        green_count = total_points - red_count - yellow_count
        
        st.metric("Total Data Points", total_points)
        st.metric("Critical Potholes (Red)", red_count, delta_color="inverse")
        st.metric("Rough Patches (Yellow)", yellow_count, delta_color="off")
        
        st.subheader("Road Condition Breakdown")
        
        st.progress(red_count / total_points, text=f"Red Zones ({red_count/total_points:.1%})")
        st.progress(yellow_count / total_points, text=f"Yellow Zones ({yellow_count/total_points:.1%})")
        st.progress(green_count / total_points, text=f"Green Zones ({green_count/total_points:.1%})")
        
        st.subheader("Detection Method Summary")
        st.markdown(
            """
            The system leverages crowd-sourced smartphone sensor data[cite: 414].
            Detection is achieved through **vibration-based analysis** using the accelerometer[cite: 43, 421].
            This process involves:
            1. Buffering data in time-series windows (e.g., 1s window)[cite: 153].
            2. Extracting features (e.g., Standard Deviation)[cite: 164].
            3. Applying ML logic to classify severity (Red/Yellow/Green)[cite: 184].
            """
        )

if __name__ == "__main__":
    main()