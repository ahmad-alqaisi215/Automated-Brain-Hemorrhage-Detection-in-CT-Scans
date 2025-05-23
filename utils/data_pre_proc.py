import shutil
import os
import pydicom
import pandas as pd
import cv2
import numpy as np
import base64
import torch

from PIL import Image
from io import BytesIO
from utils.config import (IMG_DIR, UPLOAD_DIR, AUTOCROP, SIZE, LABEL_COLS,
                          TRANSPOSEVAL, HFLIPVAL, MEAN_IMG, STD_IMG)
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, HorizontalFlip, Transpose, Normalize


class PatientLevelEmbeddingDataset:
    def __init__(self, df, mat, labels=LABEL_COLS):
        self.data = df
        self.mat = mat
        self.labels = labels
        self.patients = df.SliceID.unique()
        self.data = self.data.set_index('SliceID')

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patidx = self.patients[idx]
        patdf = self.data.loc[[patidx]].sort_values(by='seq')
        patemb = self.mat[patdf['embidx'].values]

        patdeltalag = np.zeros(patemb.shape)
        patdeltalead = np.zeros(patemb.shape)
        patdeltalag[1:] = patemb[1:]-patemb[:-1]
        patdeltalead[:-1] = patemb[:-1]-patemb[1:]

        patemb = np.concatenate((patemb, patdeltalag, patdeltalead), -1)

        ids = torch.tensor(patdf['embidx'].values)

        if self.labels:
            labels = torch.tensor(patdf[LABEL_COLS].values)
            return {'emb': patemb, 'embidx': ids, 'labels': labels}
        else:
            return {'emb': patemb, 'embidx': ids}


class IntracranialDataset(Dataset):
    def __init__(self, df, path, labels, transform=None):
        self.path = path
        self.data = df
        self.transform = transform
        self.labels = labels
        self.crop = AUTOCROP

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(
            IMG_DIR, self.data.loc[idx, 'SOPInstanceUID'] + '.jpg')
        img = cv2.imread(img_name)

        if img is None:
            raise FileNotFoundError(
                f"Failed to load image at index {idx}: {img_name}")

        if self.crop:
            try:
                img = autocrop(img, threshold=0)
            except:
                raise FileNotFoundError(
                    f"Failed to autocrop image at index {idx}: {img_name}")

        img = cv2.resize(img, (SIZE, SIZE))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        if self.labels:
            labels = torch.tensor(
                self.data.loc[idx, LABEL_COLS])
            return {'image': img, 'labels': labels}
        else:
            return {'image': img}


def remove_dir_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# ---------------- Meta Extractor --------------#


def generate_df(base, files):
    dcms_di = {}

    for filename in files:
        path = os.path.join(base,  filename)
        dcm = pydicom.dcmread(path, force=True)
        all_keywords = dcm.dir()
        ignored = ['Rows', 'Columns', 'PixelData']

        for name in all_keywords:
            if name in ignored:
                continue

            if name not in dcms_di:
                dcms_di[name] = []

            dcms_di[name].append(dcm[name].value)

    df = pd.DataFrame(dcms_di)

    return df


def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {attr: cast(getattr(dicom, attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}


def rescale_image(image, slope, intercept):
    return image * slope + intercept


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def get_dicom_meta(dicom):
    return {
        'PatientID': dicom.PatientID,  # can be grouped (20-548)
        'StudyInstanceUID': dicom.StudyInstanceUID,  # can be grouped (20-60)
        'SeriesInstanceUID': dicom.SeriesInstanceUID,  # can be grouped (20-60)
        'WindowWidth': get_dicom_value(dicom.WindowWidth),
        'WindowCenter': get_dicom_value(dicom.WindowCenter),
        'RescaleIntercept': float(dicom.RescaleIntercept),
        'RescaleSlope': float(dicom.RescaleSlope),  # all same (1.0)
    }


def apply_window_policy(image):

    image1 = apply_window(image, 40, 80)  # brain
    image2 = apply_window(image, 80, 200)  # subdural
    image3 = apply_window(image, 40, 380)  # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1, 2, 0)

    return image


def convert_dicom_to_jpg(name, rescaledict):
    dicom = pydicom.dcmread(os.path.join(UPLOAD_DIR, name), force=True)
    imgnm = name.replace('.dcm', '')
    image = dicom.pixel_array
    image = rescale_image(
        image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
    image = apply_window_policy(image)
    image -= image.min((0, 1))
    image = (255*image).astype(np.uint8)
    cv2.imwrite(os.path.join(IMG_DIR, imgnm)+'.jpg', image)


def convert_img_to_base64(img):
    buffered = BytesIO()
    Image.fromarray(img).save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_img = base64.b64encode(img_bytes).decode()
    return base64_img


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """

    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]

    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]

    sqside = max(image.shape)

    imageout = np.zeros((sqside, sqside, 3), dtype='uint8')
    imageout[:image.shape[0], :image.shape[1], :] = image.copy()

    return imageout


def get_img_transformer():
    return Compose([
        HorizontalFlip(p=HFLIPVAL),
        Transpose(p=TRANSPOSEVAL),
        Normalize(mean=MEAN_IMG, std=STD_IMG, max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])


def loademb(emb_no=0):
    return np.load(os.path.join(UPLOAD_DIR, f'emb{emb_no}.npz'))['arr_0']


def collatefn(batch):
    maxlen = max([l['emb'].shape[0] for l in batch])
    embdim = batch[0]['emb'].shape[1]
    withlabel = 'labels' in batch[0]
    if withlabel:
        labdim = batch[0]['labels'].shape[1]

    for b in batch:
        masklen = maxlen-len(b['emb'])
        b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
        b['embidx'] = torch.cat(
            (torch.ones((masklen), dtype=torch.long)*-1, b['embidx']))
        b['mask'] = np.ones((maxlen))
        b['mask'][:masklen] = 0.
        if withlabel:
            b['labels'] = np.vstack(
                (np.zeros((maxlen-len(b['labels']), labdim)), b['labels']))

    outbatch = {'emb': torch.tensor(np.vstack([np.expand_dims(b['emb'], 0)
                                               for b in batch])).float()}
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0)
                                               for b in batch])).float()
    outbatch['embidx'] = torch.tensor(np.vstack([np.expand_dims(b['embidx'], 0)
                                                for b in batch])).float()
    if withlabel:
        outbatch['labels'] = torch.tensor(
            np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    return outbatch


def long_to_wide(df):
    df_copy = df.copy()

    _ = df_copy.ID.str.split(r'(ID_[a-z|A-Z|0-9]+)_', expand=True)
    df_copy['ID'] = _.iloc[:, 1]
    df_copy['Type'] = _.iloc[:, 2]
    df_copy = df_copy.pivot(index="ID", columns="Type", values="Label")

    return df_copy
