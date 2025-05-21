import streamlit as st
import pydicom
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from utils.data_pre_proc import rescale_image, apply_window


def convert_img_to_base64(img):
    buffered = BytesIO()
    Image.fromarray(img).save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_img = base64.b64encode(img_bytes).decode()
    return base64_img

def show():
    st.title("🖥️ DCM Viewer")
    uploaded_files= st.file_uploader("Upload multiple DICOM slices", type=["dcm"], accept_multiple_files=True, key="multi")
    if uploaded_files:
        slices = []
        for file in uploaded_files:
            try:
                dcm = pydicom.dcmread(file)
                img = dcm.pixel_array.astype(float)
                img = rescale_image(img, dcm.RescaleSlope, dcm.RescaleIntercept)
                slices.append({
                    "raw_image": img,
                    "dcm": dcm,
                    "position": getattr(dcm, "ImagePositionPatient", [0])[2]
                    if hasattr(dcm, "ImagePositionPatient")
                    else dcm.get("InstanceNumber", 0)
                })
            except Exception as e:
                st.warning(f"Skipping file due to error: {e}")
                
            #sort slices
        slices= sorted(slices, key=lambda x: x['position']) 
            
        if slices:
                st.markdown("### Windowing Controls")
                
                #session state default values
                if "WC" not in st.session_state:
                    st.session_state.WC = 40
                if "WW" not in st.session_state:
                    st.session_state.WW = 80
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 Reset to Brain Window"):
                        st.session_state.WC = 40
                        st.session_state.WW = 80
                    
                        
                st.html('<div class="centered-controls">')
                col_wc, col_ww = st.columns([1, 1])
                with col_wc:
                    wc_input = st.number_input("Window Center (WC)", value=st.session_state.WC, key="wc_box")
                with col_ww:
                    ww_input = st.number_input("Window Width (WW)", value=st.session_state.WW, min_value=1, key="ww_box")
                st.html('</div>')
                
                st.session_state.WC = wc_input
                st.session_state.WW = ww_input
                
                # Slice navigation
                idx = st.session_state.get("slice_idx", 0)
                selected = slices[idx]
                
                # Apply windowing
                img_windowed = apply_window(selected["raw_image"], wc_input, ww_input)
                img_windowed -= img_windowed.min()
                img_uint8 = (img_windowed / img_windowed.max() * 255).astype(np.uint8)
                
                #display img centered inside a black div
                img_base64= convert_img_to_base64(img_uint8)
    
                st.html(f"""
                        <div style="text-align: center;">
                        <img src="data:image/png;base64,{img_base64}" width="400" style="border-radius: 8px;">
                        </div>
                        """)
                
        
                idx = st.slider("Navigate slices", 0, len(slices) - 1, 0)
                st.session_state.slice_idx = idx