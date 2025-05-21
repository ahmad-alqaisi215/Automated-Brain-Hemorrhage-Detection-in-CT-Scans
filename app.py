import streamlit as st
from utils.config import TMP_DIR, PAGE_CONFIG
from components.sidebar import render_sidebar
from utils.app_state import init_session_state
from utils.data_pre_proc import remove_dir_if_exists


# ------------------ CONFIG ------------------ #
st.set_page_config(**PAGE_CONFIG)

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
    remove_dir_if_exists(TMP_DIR)
    from app_pages.home import show

elif page == "DCM Viewer":
    remove_dir_if_exists(TMP_DIR)
    from app_pages.dcm_viewer import show

elif page == "Diagnosis":
    remove_dir_if_exists(TMP_DIR)
    from app_pages.diagnosis import show

elif page == "Brain Anatomy":
    remove_dir_if_exists(TMP_DIR)
    from app_pages.brain_anatomy import show

elif page == "About":
    remove_dir_if_exists(TMP_DIR)
    from app_pages.about import show

else:
    st.error("Unknown page selected!")

show()