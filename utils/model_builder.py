import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from utils.config import DEVICE, N_CLASSES, FEATURE_EXTRACTOR_PTH, MODELS_DIR, N_GPU, BATCH_SIZE, LABEL_COLS


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x


class SeqModel(nn.Module):
    def __init__(self, embed_size, LSTM_UNITS=64, DO = 0.3):
        super(SeqModel, self).__init__()

        self.embedding_dropout = SpatialDropout(0.0)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear = nn.Linear(LSTM_UNITS*2, len(LABEL_COLS))

    def forward(self, x, lengths=None):
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm1 = h_lstm1.to(dtype=torch.float32)
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_lstm2 = h_lstm2.to(dtype=torch.float32)

        h_conc_linear1  = F.relu(self.linear1(h_lstm1))

        h_conc_linear2  = F.relu(self.linear2(h_lstm2))

        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd

        output = self.linear(hidden)

        return output


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor.requires_grad_()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam -= cam.min()
        cam /= (cam.max() + 1e-6)

        return cam.squeeze().cpu().numpy(), output.softmax(dim=1).squeeze().detach().cpu().numpy()


def get_feature_extractor(checkpoint_no=0):
    model = torch.load(FEATURE_EXTRACTOR_PTH, weights_only=False)
    model.fc = torch.nn.Linear(2048, N_CLASSES)

    model.to(DEVICE)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(N_GPU)[::-1]), output_device=DEVICE)
    for param in model.parameters():
        param.requires_grad = False

    input_model_file = os.path.join(
        MODELS_DIR, f"model_999_epoch{checkpoint_no}_fold6.bin")
    model.load_state_dict(torch.load(input_model_file))
    model.to(DEVICE)
    model.eval()

    return model


def get_data_loader(ichdataset):
    return DataLoader(ichdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())


def make_diagnosis(ypred, imgs):
    imgls = np.array(imgs).repeat(len())
    icdls = pd.Series(LABEL_COLS * ypred.shape[0])
    yidx = ['{}_{}'.format(i,j) for i,j in zip(imgls, icdls)]
    subdf = pd.DataFrame({'ID' : yidx, 'Label': ypred.flatten()})
    return subdf