import streamlit as st


def init_session_state():
    if "page_selector" not in st.session_state:
        st.session_state.page_selector = "Home"
