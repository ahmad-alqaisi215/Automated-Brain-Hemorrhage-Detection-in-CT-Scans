import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

def show():
    st.title("🧠 Brain Anatomy")
    st.html("""
        <style>
        .ich-card {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .ich-card img {
            width: 100px;
            height: 100px;
            border-radius: 8px;
            object-fit: cover;
            border: 2px solid #ccc;
        }
        .ich-card h4 {
            margin-bottom: 5px;
        }
        .ich-card p {
            margin: 0;
            font-size: 15px;
            color: #333;
        }
        </style>
    """)
    #glb_url = 'https://raw.githubusercontent.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/main/human-brain.glb'
    #glb_url = "https://raw.githubusercontent.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/main/plastic_skull_with_brain_pathologies_model.glb"
    #glb_url='https://raw.githubusercontent.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/main/components/brain_anatomy.glb'
    #glb_url='https://raw.githubusercontent.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/main/brain_anatomy_2.glb'
    glb_url= 'https://raw.githubusercontent.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans/main/brain_anatomy_final.glb'
    
    components.html(f"""
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

<model-viewer 
  id="brainModel"
  src="{glb_url}"
  alt="3D brain model"
  auto-rotate
  camera-controls
  style="width: 100%; height: 700px; background-color: white; border-radius: 10px;"
  exposure="1"
  shadow-intensity="1"
  environment-image="neutral"
  field-of-view="35deg"
  camera-target="0m 0m 0m">""", height=400)

    
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
        st.html("""
        <div class="ich-card" style="background-color: #ffe6e6;">
            <img src="https://i.postimg.cc/5NbvWsqZ/epi.png" />
            <div>
                <h4>1. Epidural Hemorrhage (EDH)</h4>
                <p>Bleeding between the skull and dura mater.<br>Source: Arterial, Often caused by trauma.<br>Shape: Forms a lens-shaped clot on CT.
                </p>
            </div>
        </div>
    """)
        st.html("""
        <div class="ich-card" style="background-color: #fff0cc;">
            <img src="https://i.postimg.cc/nLyjR2hS/sub.png" />
            <div>
                <h4>2. Subdural Hemorrhage (SDH)</h4>
                <p>Bleeding between the dura and the arachnoid.<br>Source: Usually from torn veins.<br>Shape: Appears crescent-shaped and spreads widely.                
                </p>
            </div>
        </div>
    """)
        st.html("""
        <div class="ich-card" style="background-color: #e6f2ff;">
            <img src="https://i.postimg.cc/rwdwB4bB/sba.png" />
            <div>
                <h4>3. Subarachnoid Hemorrhage (SAH)</h4>
                <p>Bleeding between the arachnoid and the pia mater.<br>Source: Often due to a ruptured aneurysm.<br>Shape: Tracks along the sulci and fissures
                </p>
            </div>
        </div>
    """)
        with col2:
            st.html("""
        <div class="ich-card" style="background-color: #e6ffe6;">
            <img src="https://i.postimg.cc/3NJKn0mb/iph.png" />
            <div>
                <h4>4. Intraparenchymal Hemorrhage (IPH)</h4>
                <p>Bleeding inside the brain tissue itself.<br>Source: Arterial or venous<br>Shape: Shows as a dense area in brain matter.
                </p>
            </div>
        </div>
    """)
            st.html("""
        <div class="ich-card" style="background-color: #f0e6ff;">
            <img src="https://i.postimg.cc/BQ7v7ph5/ivh.png" />
            <div>
                <h4>5. Intraventricular Hemorrhage (IVH)</h4>
                <p>Bleeding into the brain’s ventricles (fluid spaces).<br>Source: Arterial or venous<br>Shape: Conforms to ventricular shape


                </p>
            </div>
        </div>
    """)
            st.html("""
        <div class="ich-card" style="background-color: #f2f2f2;">
            <img src="https://i.postimg.cc/fRg0XDnN/normal.png" />
            <div>
                <h4>6. Normal</h4>
                <p>Normal and Healthy Brain.<br><br><br></p>
            </div>
        </div>
    """)

    st.markdown("""---""")
    st.markdown("""
### 🧬 Clinical Insight:
> Early detection of ICH on CT scans can **save lives**. This app uses a deep learning model to assist in identifying multiple types of ICH across CT slices.
""")
