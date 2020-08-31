#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import json
import librosa
import torch
import soundfile as sf
import warnings


# In[ ]:


def l1_l2_loss(y_true, y_pred, l1_weight, l2_weight):
    loss = 0

    if l1_weight != 0:
        loss += l1_weight*torch.nn.L1Loss()(y_true, y_pred)

    if l2_weight != 0:
        loss += l2_weight * torch.nn.MSELoss()(y_true, y_pred)

    return loss


# In[ ]:


def compute_receptive_field_length(stacks, dilations, filter_length, target_field_length):

    half_filter_length = (filter_length-1)/2
    length = 0
    for d in dilations:
        length += d*half_filter_length
    length = 2*length
    length = stacks * length
    length += target_field_length
    return length


# In[ ]:


def one_hot_encode(x, num_values=256):
    if isinstance(x, int):
        x = np.array([x])
    if isinstance(x, list):
        x = np.array(x)
    return np.eye(num_values, dtype='uint8')[x.astype('uint8')]


# In[ ]:


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


# In[ ]:


def binary_encode(x, max_value):
    if isinstance(x, int):
        x = np.array([x])
    if isinstance(x, list):
        x = np.array(x)
    width = np.ceil(np.log2(max_value)).astype(int)
    return (((x[:, None] & (1 << np.arange(width)))) > 0).astype(int)


# In[ ]:


def get_condition_input_encode_func(representation):
    if representation == 'binary':
        return binary_encode
    else:
        return one_hot_encode


# In[ ]:


def ensure_keys_in_dict(keys, dictionary):
    if all (key in dictionary for key in keys):
        return True
    return False


# In[ ]:


def get_subdict_from_dict(keys, dictionary):
    return dict((k, dictionary[k]) for k in keys if k in dictionary)


# In[ ]:


def load_wav(wav_path, desired_sample_rate):
    sequence, _ = librosa.load(wav_path, sr = desired_sample_rate)
    return sequence


# In[ ]:


def rms(x):
    return np.sqrt(np.mean(np.square(x), axis=-1))


# In[ ]:


def normalize(x):
    max_peak = np.max(np.abs(x))
    return x / max_peak


# In[ ]:


def get_subsequence_with_speech_indices(full_sequence):
    signal_magnitude = np.abs(full_sequence)

    chunk_length = 800

    chunks_energies = []
    for i in range(0, len(signal_magnitude), chunk_length):
        chunks_energies.append(np.mean(signal_magnitude[i:i + chunk_length]))

    threshold = np.max(chunks_energies) * .1

    onset_chunk_i = 0
    for i in range(0, len(chunks_energies)):
        if chunks_energies[i] >= threshold:
            onset_chunk_i = i
            break

    termination_chunk_i = len(chunks_energies)
    for i in range(len(chunks_energies) - 1, 0, -1):
        if chunks_energies[i] >= threshold:
            termination_chunk_i = i
            break

    num_pad_chunks = 4
    onset_chunk_i = np.max((0, onset_chunk_i - num_pad_chunks))
    termination_chunk_i = np.min((len(chunks_energies), termination_chunk_i + num_pad_chunks))

    return [onset_chunk_i*chunk_length, (termination_chunk_i+1)*chunk_length]


# In[ ]:


def dir_contains_files(path):
    file_list = os.listdir(path)
    if not len(file_list)==0:
        return True
    else: 
        return False

def snr_db(rms_amplitude_A, rms_amplitude_B):
    return 20.0*np.log10(rms_amplitude_A/rms_amplitude_B)

def write_wav(x, filename, sample_rate):

    if type(x) != np.ndarray:
        x = np.array(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf.write(filename, x, sample_rate)


