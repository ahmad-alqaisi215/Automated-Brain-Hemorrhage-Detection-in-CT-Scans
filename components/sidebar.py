import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("# Navigation")

        if st.button(" Home"):
            st.session_state.page_selector = "Home"
            st.rerun()
        if st.button(" Brain Anatomy"):
            st.session_state.page_selector = "Brain Anatoym"
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
