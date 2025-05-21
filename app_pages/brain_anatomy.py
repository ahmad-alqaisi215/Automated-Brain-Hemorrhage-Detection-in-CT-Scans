import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

def show():
    st.title("🧠 Brain Anatomy")
    glb_url = 'https://raw.githubusercontent.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/main/human-brain.glb'


# Display the 3D model using <model-viewer>
    components.html(f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

    <model-viewer
        src="{glb_url}"
        alt="3D brain model"
        auto-rotate
        camera-controls
        style="width: 100%; height: 350px; background-color: white; border-radius: 10px;"
        exposure="1"
        shadow-intensity="1">
    </model-viewer>
""", height=520)
    
    # Section: What is ICH?
    st.markdown("## 🩸 What is Intracranial Hemorrhage (ICH)?")
    st.info(
    "Intracranial Hemorrhage (ICH) is bleeding that occurs inside the skull, often due to trauma, high blood pressure, or ruptured blood vessels. "
    "It can compress brain tissue and is considered a medical emergency that requires rapid diagnosis and intervention."
)

# Section: ICH Types (creative 2-column layout)
    st.markdown("## 🧠 Types of ICH")

    col1, col2 = st.columns(2)

    with col1:
        st.success("**1. Epidural Hemorrhage (EDH)**\n\nBleeding between the skull and dura mater. Often caused by trauma.")
        st.warning("**2. Subdural Hemorrhage (SDH)**\n\nOccurs between the dura and arachnoid mater. Common in elderly or after head injury.")
        st.error("**3. Subarachnoid Hemorrhage (SAH)**\n\nBleeding in the space around the brain. Often from ruptured aneurysms.")

    with col2:
        st.info("**4. Intraparenchymal Hemorrhage (IPH)**\n\nBleeding inside the brain tissue. Often due to high blood pressure or stroke.")
        st.success("**5. Intraventricular Hemorrhage (IVH)**\n\nBleeding into the brain's ventricular system. Common in premature infants and trauma.")

    # Optional fun facts or callout
    st.markdown("""---""")
    st.markdown("""
### 🧬 Clinical Insight:
> Early detection of ICH on CT scans can **save lives**. This app uses a deep learning model to assist in identifying multiple types of ICH across CT slices.
""")

