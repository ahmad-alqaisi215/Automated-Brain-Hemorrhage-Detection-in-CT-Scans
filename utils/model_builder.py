import torch
import os

from utils.config import DEVICE, N_CLASSES, FEATURE_EXTRACTOR_PTH, MODELS_DIR, N_GPU


def get_feature_extractor(checkpoint_no=0):
    model = torch.load(FEATURE_EXTRACTOR_PTH, weights_only=False)
    model.fc = torch.nn.Linear(2048, N_CLASSES)

    model.to(DEVICE)
    model = torch.nn.DataParallel(model, device_ids=list(range(N_GPU)[::-1]), output_device=DEVICE)
    for param in model.parameters():
        param.requires_grad = False

    input_model_file = os.path.join(MODELS_DIR, f"model_999_epoch{checkpoint_no}_fold6.bin")
    model.load_state_dict(torch.load(input_model_file))
    model.to(DEVICE)
    model.eval()

    return model