
import os
import pydicom
import pandas as pd


def generate_df(base, files):
    dcms_di = {}

    for filename in files:
        path = os.path.join(base,  filename)
        dcm = pydicom.dcmread(path)
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
