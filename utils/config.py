import os


PAGE_CONFIG = dict (
    page_title="ICH Detection Assistant", 
    page_icon="🧠", 
    layout="wide"
    )

MEAN_IMG = [0.22363983, 0.18190407, 0.2523437]
STD_IMG = [0.32451536, 0.2956294,  0.31335256]
ORIG_IMG_SIZE = 512
TMP_DIR = 'tmp'
UPLOAD_DIR = os.path.join(TMP_DIR, 'uploads')