
import os
import pydicom
import pandas as pd
import cv2
import numpy as np

from utils.config import IMG_DIR, UPLOAD_DIR

#---------------- Meta Extractor --------------#
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
    return {attr:cast(getattr(dicom,attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}


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
        'PatientID': dicom.PatientID, # can be grouped (20-548)
        'StudyInstanceUID': dicom.StudyInstanceUID, # can be grouped (20-60)
        'SeriesInstanceUID': dicom.SeriesInstanceUID, # can be grouped (20-60)
        'WindowWidth': get_dicom_value(dicom.WindowWidth),
        'WindowCenter': get_dicom_value(dicom.WindowCenter),
        'RescaleIntercept': float(dicom.RescaleIntercept),
        'RescaleSlope': float(dicom.RescaleSlope), # all same (1.0)
    }


def apply_window_policy(image):

    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image

def convert_dicom_to_jpg(name, rescaledict):
    dicom = pydicom.dcmread(os.path.join(UPLOAD_DIR, name), force=True)
    imgnm = name.replace('.dcm', '')
    image = dicom.pixel_array
    image = rescale_image(image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
    image = apply_window_policy(image)
    image -= image.min((0,1))
    image = (255*image).astype(np.uint8)
    cv2.imwrite(os.path.join(IMG_DIR, imgnm)+'.jpg', image)
