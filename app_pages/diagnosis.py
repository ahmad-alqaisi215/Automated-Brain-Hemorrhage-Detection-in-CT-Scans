import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
import gc
import cv2
import ast

from utils.config import UPLOAD_DIR, IMG_DIR, DEVICE, BATCH_SIZE, LSTM_UNITS, SEQ_MODEL_PTH, N_CLASSES, N_GPU, MODELS_DIR, N_BAGS
from utils.data_pre_proc import (generate_df, convert_dicom_to_jpg, IntracranialDataset, get_img_transformer, 
                                 loademb, PatientLevelEmbeddingDataset, collatefn, bagged_diagnosis, convert_img_to_base64)
from utils.model_builder import get_feature_extractor, get_data_loader, GradCAM, SeqModel, predict, make_diagnosis, Identity
from torch.utils.data import DataLoader

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
        if "files_uploaded" not in st.session_state:
            st.session_state.files_uploaded = False

        if uploaded_files and not st.session_state.files_uploaded:
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
            st.session_state.files_uploaded = True

        if "slices_processed" not in st.session_state:
            st.session_state.slices_processed = False

        if not st.session_state.slices_processed:
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
            st.session_state.slices_processed = True

        ichdataset = IntracranialDataset(
            df=pd.read_csv(os.path.join(UPLOAD_DIR, 'metadata.csv')),
            path=IMG_DIR,
            transform=get_img_transformer(),
            labels=False
        )

        if "embeddings_extracted" not in st.session_state:
            st.session_state.embeddings_extracted = False

        if not st.session_state.embeddings_extracted:
            loader = get_data_loader(ichdataset)
            extract_emb = st.progress(0.0, text="Embedding Extraction...")

            total_models = 1
            step = 0

            for i in range(total_models):
                model = get_feature_extractor(i)

                model.module.fc = Identity()
                model.eval()

                ls = []

                for j, batch in enumerate(loader):
                    inputs = batch["image"].to(DEVICE, dtype=torch.float)
                    out = model(inputs)
                    ls.append(out.detach().cpu().numpy())

                    if j % total_models == 0:
                        step += BATCH_SIZE
                        progress = step / len(ichdataset)

                    extract_emb.progress(
                        min(progress, 1.0),
                        text=f'Embedding Extraction ({min(step, len(ichdataset))}/{len(ichdataset)})...'
                    )

                outemb = np.concatenate(ls, 0).astype(np.float32)
                np.savez_compressed(os.path.join(UPLOAD_DIR, f'emb{i}'), outemb)
                gc.collect()

            extract_emb.progress(
                1.0, text=f'Embedding Extraction ({len(ichdataset)}/{len(ichdataset)})...')

            st.success(f"{len(ichdataset)} CT Slice(s) Embeddings Extracted!")

            model = get_feature_extractor()
            gradcam = GradCAM(model, target_layer=model.module.layer4[-1])

            for i, batch in enumerate(loader):
                inputs = batch["image"].to(DEVICE)
                ids = batch["id"][0]

                for j in range(inputs.shape[0]):
                    input_tensor = inputs[j].unsqueeze(0)
                    cam, _ = gradcam.generate(input_tensor)

                    img = input_tensor.squeeze().detach().cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))

                    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                    img = np.uint8(255 * img)

                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

                    out_path = os.path.join(UPLOAD_DIR, f"{ids[j]}.jpg")
                    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            dcms_df['SliceID'] = dcms_df[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(
                lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
            
            poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
            dcms_df[poscols] = pd.DataFrame(dcms_df['ImagePositionPatient']
                                    .apply(lambda x: list(map(float, x))).tolist())
            
            dcms_df = dcms_df.sort_values(
                ['SliceID']+poscols)[['PatientID', 'SliceID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
            dcms_df['seq'] = (dcms_df.groupby(['SliceID']).cumcount() + 1)

            keepcols = ['PatientID', 'SliceID', 'SOPInstanceUID', 'seq']
            dcms_df = dcms_df[keepcols]

            dcms_df.columns = dcms_df.columns = ['PatientID', 'SliceID', 'Image', 'seq']

            dcms_df_seq = loader.dataset.data
            dcms_df_seq['Image'] = dcms_df_seq['SOPInstanceUID']
            dcms_df_seq['embidx'] = range(dcms_df_seq.shape[0])

            dcms_df_seq = dcms_df_seq.merge(dcms_df, on='Image')
            lstmypredls = []

            for i in range(total_models):
                dcms_df_emb = [loademb(i)]
                dcms_df_emb = sum(dcms_df_emb)/len(dcms_df_emb)

                dcm_seq_dataset = PatientLevelEmbeddingDataset(dcms_df_seq, dcms_df_emb, labels=False)
                dcm_seq_loader = DataLoader(dcm_seq_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), collate_fn=collatefn)

                model = SeqModel(embed_size=LSTM_UNITS*3, LSTM_UNITS=LSTM_UNITS, DO=0.0)
                model.to(DEVICE)

                # model = torch.nn.DataParallel(model, device_ids=list(
                #     range(N_GPU)[::-1]), output_device=DEVICE)
                
                for param in model.parameters():
                    param.requires_grad = False

                input_model_file = os.path.join(
                    MODELS_DIR, f"lstm_gepoch{i}_lstmepoch11_fold6.bin")
                model.load_state_dict(torch.load(input_model_file))
                model.to(DEVICE)
                model.eval()

                ypredls = []

                ypred, imgdcm = predict(dcm_seq_loader, model)
                ypredls.append(ypred)

                ypred = sum(ypredls[-N_BAGS:])/len(ypredls[-N_BAGS:])
                yout = make_diagnosis(ypred, imgdcm)
                
                lstmypredls.append(yout.set_index('ID'))

            final_diagnosis = bagged_diagnosis(lstmypredls)
            final_diagnosis.to_csv(os.path.join(UPLOAD_DIR, 'final_diagnosis.csv'), index=False)
            # st.subheader("📋 Final Diagnosis Table")
            # st.dataframe(final_diagnosis)

            st.session_state.embeddings_extracted = True

        dcm_meta = pd.read_csv(os.path.join(UPLOAD_DIR, 'metadata.csv'))

        # ----------------- Load and Sort (only once) ----------------- #
        if "viewer_loaded" not in st.session_state:
            st.session_state.viewer_loaded = False

        if not st.session_state.viewer_loaded:
            dcm_meta = pd.read_csv(os.path.join(UPLOAD_DIR, 'metadata.csv'))

            dcm_meta['z'] = dcm_meta['ImagePositionPatient'].apply(
                lambda x: ast.literal_eval(x)[-1] if isinstance(x, str) else 0
            )

            dcm_meta.sort_values(by='z', inplace=True)
            dcm_meta.reset_index(drop=True, inplace=True)

            st.session_state.dcm_meta = dcm_meta
            st.session_state.viewer_loaded = True
        else:
            dcm_meta = st.session_state.dcm_meta

        # ----------------- Display ----------------- #
        st.markdown("### 🧠 CT Scan Slice Viewer")

        # Load images using OpenCV
        idx = st.session_state.get("slice_idx", 0)  # safe default if not initialized

        orig_img = cv2.imread(os.path.join(IMG_DIR, dcm_meta.loc[idx, 'SOPInstanceUID']) + '.jpg', cv2.IMREAD_GRAYSCALE)
        grad_img = cv2.imread(os.path.join(UPLOAD_DIR, dcm_meta.loc[idx, 'SOPInstanceUID']) + '.jpg', cv2.IMREAD_COLOR)

        # Convert grayscale
        gray_base64 = convert_img_to_base64(orig_img)
        grad_base64 = convert_img_to_base64(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original CT Image**")
            st.markdown(f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{gray_base64}" width="400" style="border-radius: 8px;" />
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("**Potential Hemorrhage (AI Interpretation for Clinical Review)**")
            st.markdown(f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{grad_base64}" width="400" style="border-radius: 8px;" />
            </div>
            """, unsafe_allow_html=True)

        st.slider("Navigate Slices", 0, len(dcm_meta) - 1, key="slice_idx")

