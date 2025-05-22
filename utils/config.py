import os
import torch


PAGE_CONFIG = dict (
    page_title="ICH Detection Assistant", 
    page_icon="🧠", 
    layout="wide"
    )

MEAN_IMG = [0.22363983, 0.18190407, 0.2523437]
STD_IMG = [0.32451536, 0.2956294,  0.31335256]
ORIG_IMG_SIZE = 512

TMP_DIR = 'tmp'
MODELS_DIR = 'models'
UPLOAD_DIR = os.path.join(TMP_DIR, 'uploads')
IMG_DIR = os.path.join(TMP_DIR, 'image')
FEATURE_EXTRACTOR_PTH = os.path.join(MODELS_DIR, 'resnext101_32x8d_wsl_checkpoint.pth')

N_CLASSES = 6

DEVICE = torch.device('cuda')
N_GPU = torch.cuda.device_count()