import streamlit as st
from components.sidebar import render_sidebar
from utils.app_state import init_session_state

# ------------------ CONFIG ------------------ #
st.set_page_config(
    page_title="ICH Detection Assistant", 
    page_icon="🧠", 
    layout="wide"
    )

# ------------------ SESSION SETUP ------------------ #
init_session_state()

# ------------------ CUSTOM CSS ------------------ #
with open("static/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------ #
render_sidebar()

# ------------------ ROUTING ------------------ #
page = st.session_state.page_selector

if page == "Home":
    from pages.home import show
elif page == "DCM Viewer":
    from pages.dcm_viewer import show
elif page == "Diagnosis":
    from pages.diagnosis import show
elif page == "Brain Anatomy":
    from pages.brain_anatomy import show
elif page == "About":
    from pages.about import show
else:
    st.error("Unknown page selected!")

show()