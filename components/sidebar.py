import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

def render_sidebar():
    with st.sidebar:
        glb_url = 'https://raw.githubusercontent.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/main/human-brain.glb'
        
        components.html(f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

    <model-viewer
        src="{glb_url}"
        alt="3D brain model"
        auto-rotate
        camera-controls
        style="width: 100px; height: 100px; background: transparent; border-radius: 10px;"
        exposure="1"
        shadow-intensity="1">
    </model-viewer>
""", height=100)
        
        st.markdown("<div class='navbar'>", unsafe_allow_html=True)
        st.markdown("# Navigation")

        if st.button(" Home"):
            st.session_state.page_selector = "Home"
            st.rerun()
        if st.button(" Brain Anatomy"):
            st.session_state.page_selector = "Brain Anatomy"
            st.rerun()
        if st.button(" DCM Viewer"):
            st.session_state.page_selector = "DCM Viewer"
            st.rerun()
        if st.button(" Diagnosis"):
            st.session_state.page_selector = "Diagnosis"
            st.rerun()
        if st.button(" About"):
            st.session_state.page_selector = "About"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
