import streamlit as st


def show():
    st.title("👋 Welcome to the ICH Detection Assistant")

    st.html("""
                <div style='text-align: left; font-size: 1em; color: #333; background-color: #f9f9f9; padding: 1em; border-radius: 8px;'>
                    <p>
                        A clinically-informed AI tool to assist healthcare professionals in identifying intracranial hemorrhages from CT scans.
                    </p>

                    <p style="font-style: italic; margin-top: 1em; font-size: 1.5em;">
                        ⚠️ <strong>Note:</strong> This tool supports medical decisions but does not replace a doctor’s judgment.
                        Always consult a qualified healthcare professional for diagnosis and treatment.
                    </p>

                    <hr style="border: 1px solid #ccc; margin-top: 1.5em;" />
                </div>
    """)
