import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

def show():
    st.title("🧠 Brain Anatomy")
    model_path = Path("C:/Users/SarahAl2203.SARAHALLAPTOP/Desktop/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/human-brain.glb").as_posix()

# Display the 3D model using <model-viewer>
    components.html(f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

    <model-viewer
        src="file:///{model_path}"
        alt="3D brain model"
        auto-rotate
        camera-controls
        style="width: 100%; height: 500px; background-color: black; border-radius: 10px;"
        exposure="1"
        shadow-intensity="1">
    </model-viewer>
""", height=520)
