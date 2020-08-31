#!/usr/bin/env python
# coding: utf-8


# In[2]:



import sys
import logging
import json
import os
import data.wavenet.models as models
import data.wavenet.dataset as datasets
import data.wavenet.util as util
import data.wavenet.denoise as denoise
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# In[4]:


sys.setrecursionlimit(50000)
"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s%(name)s:%(message)s')
file_handler = logging.FileHandler('/home/sauravpathak/data/mains.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
"""

# In[5]:


def get_arguments():
    
    config='/home/sauravpathak/data/wavenet/config.json'
    mode='inference'
    load_checkpoint='/home/sauravpathak/data/NSDTSEA/checkpoints/config1_epoch0073.pth'
    condition_value=0
    batch_size=4
    one_shot=False
    clean_input_path='/home/sauravpathak/data/NSDTSEA/clean_speech/'
    noisy_input_path='/home/sauravpathak/data/NSDTSEA/noisy_speech/'
    print_model_summary=False
    target_field_length=None
    
    options = { 'config' : config,
                'mode' : mode,
                'load_checkpoint' : load_checkpoint,
                'condition_value' : condition_value,
                'batch_size' : batch_size,
                'one_shot' : one_shot,
                'clean_input_path' : clean_input_path,
                'noisy_input_path' : noisy_input_path,
                'print_model_summary' : print_model_summary,
                'target_field_length' : target_field_length
              }

    return options


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)


def get_dataset(config, model):

    if config['dataset']['type'] == 'vctk+demand':
        return datasets.VCTKAndDEMANDDataset(config, model).load_dataset()
    elif config['dataset']['type'] == 'nsdtsea':
        return datasets.NSDTSEADataset(config, model).load_dataset()


def training(config, args):

    # Instantiate Model
    model = models.DenoisingWavenet(config)
    print('model loaded...')
    dataset = get_dataset(config, model)
    print('Dataset loaded...')

    num_train_samples = config['training']['num_train_samples']
    num_test_samples = config['training']['num_test_samples']
    train_set_generator = dataset.get_random_batch_generator('train')
    test_set_generator = dataset.get_random_batch_generator('test')
    
    train_set_iterator = datasets.denoising_dataset(train_set_generator)
    test_set_iterator = datasets.denoising_dataset(test_set_generator)
    
    train_loader = DataLoader(train_set_iterator, batch_size=None)
    valid_loader = DataLoader(test_set_iterator, batch_size=None)

    dataloader = {'train_loader':train_loader, 'valid_loader':valid_loader}
    training_config = models.TrainingConfig(model, dataloader, config)
    training_config.setup_model()
    print('model setup done...')
    training_config.train(num_train_samples, num_test_samples)
    print('model training done...')


def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)
            break
        j += 1
    return output_folder_path


def inference(config, args):

    if args['batch_size'] is not None:
        batch_size = int(args['batch_size'])
    else:
        batch_size = config['training']['batch_size']

    if args['target_field_length'] is not None:
        args['target_field_length'] = int(args['target_field_length'])

    if not bool(args['one_shot']):
        model = models.DenoisingWavenet(config, target_field_length=args['target_field_length'])
        print('Performing inference..')
    else:
        print('Performing one-shot inference..')

    samples_folder_path = os.path.join(config['training']['path'], 'samples')
    output_folder_path = get_valid_output_folder_path(samples_folder_path)

    #If input_path is a single wav file, then set filenames to single element with wav filename
    if args['noisy_input_path'].endswith('.wav'):
        filenames = [args['noisy_input_path'].rsplit('/', 1)[-1]]
        args['noisy_input_path'] = args['noisy_input_path'].rsplit('/', 1)[0] + '/'
        if args['clean_input_path'] is not None:
            args['clean_input_path'] = args['clean_input_path'].rsplit('/', 1)[0] + '/'
    else:
        if not args['noisy_input_path'].endswith('/'):
            args['noisy_input_path'] += '/'
        filenames = [filename for filename in os.listdir(args['noisy_input_path']) if filename.endswith('.wav')]

    clean_input = None
    for filename in filenames:
        noisy_input = util.load_wav(args['noisy_input_path'] + filename, config['dataset']['sample_rate'])
        if args['clean_input_path'] is not None:
            clean_input = util.load_wav(os.path.join(args['clean_input_path'], filename), config['dataset']['sample_rate'])

        inputs = {'noisy': noisy_input, 'clean': clean_input}
        print(len(noisy_input), len(clean_input))

        output_filename_prefix = filename[0:-4] + '_'

        if config['model']['condition_encoding'] == 'one_hot':
            condition_input = util.one_hot_encode(int(args['condition_value']), 29)[0]
        else:
            condition_input = util.binary_encode(int(args['condition_value']), 29)[0]

        if bool(args['one_shot']):
            if len(inputs['noisy']) % 2 == 0:  # If input length is even, remove one sample
                inputs['noisy'] = inputs['noisy'][:-1]
                if inputs['clean'] is not None:
                    inputs['clean'] = inputs['clean'][:-1]
            model = models.DenoisingWavenet(config, input_length=len(inputs['noisy']))

        print("Denoising: " + filename)
        
        predict_config = models.PredictConfig(model, args['load_checkpoint'])
        denoise.denoise_sample(predict_config, inputs, condition_input, batch_size, output_filename_prefix,
                                            config['dataset']['sample_rate'], output_folder_path)

def main():

    args = get_arguments()
    config = load_config(args['config'])

    if args['mode'] == 'training':
        training(config, args)
    elif args['mode'] == 'inference':
        inference(config, args)


if __name__ == "__main__":
    main()


# In[5]:




