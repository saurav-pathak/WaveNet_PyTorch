# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Layers.py

import torch

def expand_dims(x, axes):
    
    if axes == 1:
        x = x.unsqueeze_(1)
    if axes == -1:
        x = x.unsqueeze_(-1)
    return x


def slicing(x, slice_idx, axes):
    if axes == 1:
        return x[:,slice_idx,:]
    if axes == 2:
        return x[:,:,slice_idx]
