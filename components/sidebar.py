import streamlit as st

def render_sidebar():
    st.markdown("# Navigation")

    if st.button(" Home"):
        st.session_state.page_selector = "Home"
        st.rerun()
    if st.button(" DCM Viewer"):
        st.session_state.page_selector = "DCM Viewer"
        st.rerun()
    if st.button(" Diagnosis"):
        st.session_state.page_selector = "Diagnosis"
        st.rerun()
    if st.button(" Report"):
        st.session_state.page_selector = "Report"
        st.rerun()
    if st.button(" About"):
        st.session_state.page_selector = "About"
        st.rerun()
