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
