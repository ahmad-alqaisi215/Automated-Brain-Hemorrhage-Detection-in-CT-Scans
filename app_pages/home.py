import streamlit as st


def show():
    st.title("👋 Welcome to the ICH Detection Assistant")

    st.html("""
                <div style='text-align: left; font-size: 1em; color: #333; background-color: #f9f9f9; padding: 1em; border-radius: 8px;'>
                    <p>
                        A clinically-informed AI tool to assist healthcare professionals in identifying intracranial hemorrhages from CT scans.
                    </p>
                </div>
    """)
    st.html("""
            <div style="background-color:#fff4cc; padding:20px; border-radius:10px;">
<p style="font-style: italic; margin-top: 1em; font-size: 1.5em;">
            ⚠️ <strong>Note:</strong> This tool supports medical decisions but does not replace a doctor’s judgment.
            Always consult a qualified healthcare professional for diagnosis and treatment.
            </p>
            </div>
   """)
    st.markdown("## 🩺 How to Use This Tool: ")
    st.html("""
        <ol style="text-align: left; font-size: 20px;">
        
        <li><strong>Upload a patient's CT scan</strong></li>
        <li><strong>Run the AI model</strong></li>
        <li><strong>Download report for documentation</strong></li>
        
        </ol>
        """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Tool Features")
    cols = st.columns(3)
    cols[0].success("✔️ Multi-label Detection")
    cols[1].info("📂 DICOM Support")
    cols[2].warning("📈 Visual AI Insights")


