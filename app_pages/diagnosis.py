import os
import streamlit as st

from utils.config import UPLOAD_DIR, IMG_DIR
from utils.data_pre_proc import generate_df, convert_dicom_to_jpg


def show():
    st.title("📤 Upload CT-Scan")

    uploaded_files = st.file_uploader(
        label="📤 Upload One or More **DICOM** Files (Each with a Single CT Slice)",
        type=["dcm"],
        accept_multiple_files=True,
        key="single",
        help="You can upload multiple .dcm files, but each file must represent a single-slice CT scan. Multi-frame DICOMs are not supported yet."
    )

    if uploaded_files:
        for file in uploaded_files:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) uploaded!")

        dcm_files = os.listdir(UPLOAD_DIR)
        dcms_df = generate_df(UPLOAD_DIR, dcm_files)
        dcms_df.to_csv(os.path.join(UPLOAD_DIR, 'metadata.csv'), index=False)

        rescaledict = dcms_df.set_index('SOPInstanceUID')[['RescaleSlope', 'RescaleIntercept']].to_dict()

        os.makedirs(IMG_DIR, exist_ok=True)
        for name in os.listdir(UPLOAD_DIR):
            if 'csv' in name:
                continue

            convert_dicom_to_jpg(name, rescaledict)

