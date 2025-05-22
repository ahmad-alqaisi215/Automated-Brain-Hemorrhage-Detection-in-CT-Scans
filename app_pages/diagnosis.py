import os
import streamlit as st
import time
import torch
import pandas as pd
import numpy as np
import gc
from utils.config import UPLOAD_DIR, IMG_DIR, DEVICE, BATCH_SIZE
from utils.data_pre_proc import generate_df, convert_dicom_to_jpg, IntracranialDataset, get_img_transformer
from utils.model_builder import get_feature_extractor, get_data_loader


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
        st.success(f"{len(uploaded_files)} file(s) uploaded!")
        meta_progress = st.progress(0, text=f"Files Analysis...")
        for i, file in enumerate(uploaded_files):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            meta_progress.progress(
                (i + 1)/len(uploaded_files), text=f'Files Analysis ({i + 1}/{len(uploaded_files)})...')
        
        st.success(f"{len(uploaded_files)} file(s) Analyzed!")

        slice_prepration = st.progress(0, text=f"Processing CT Slices...")

        dcm_files = os.listdir(UPLOAD_DIR)
        dcms_df = generate_df(UPLOAD_DIR, dcm_files)
        dcms_df.to_csv(os.path.join(UPLOAD_DIR, 'metadata.csv'), index=False)

        rescaledict = dcms_df.set_index('SOPInstanceUID')[
            ['RescaleSlope', 'RescaleIntercept']].to_dict()

        os.makedirs(IMG_DIR, exist_ok=True)

        for i, name in enumerate(dcm_files):
            slice_prepration.progress(
                (i + 1)/len(dcm_files),
                text=f'Processing CT Slices ({i + 1}/{len(dcm_files)})...'
            )

            convert_dicom_to_jpg(name, rescaledict)
            
        st.success(f"{len(uploaded_files)} CT Slice(s) Processed!")
        ichdataset = IntracranialDataset(
            df=pd.read_csv(os.path.join(UPLOAD_DIR, 'metadata.csv')),
            path=IMG_DIR,
            transform=get_img_transformer(),
            labels=False
        )
        
        loader = get_data_loader(ichdataset)
        model = get_feature_extractor()
        ls = []
        extract_emb = st.progress(0.0, text="Embedding Extraction...")

        total = len(loader)
        for i, batch in enumerate(loader, start=1):
            inputs = batch["image"].to(DEVICE, dtype=torch.float)
            out = model(inputs)
            ls.append(out.detach().cpu().numpy())
            progress = i / total
            extract_emb.progress(progress, text=f'Embedding Extraction ({min(len(ichdataset), i * BATCH_SIZE)}/{len(ichdataset)})...')
            
        outemb = np.concatenate(ls, 0).astype(np.float32)
        np.savez_compressed(os.path.join(UPLOAD_DIR, 'emb'), outemb)

        gc.collect()
        st.success(f"{len(ichdataset)} CT Slice(s) Embeddings Extracted!")
        st.session_state.embeddings_extracted = True

