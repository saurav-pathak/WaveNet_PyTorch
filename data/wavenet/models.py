#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch.nn.functional as F
import data.wavenet.util as util
import data.wavenet.layers as layers
import os
import logging
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm


# In[ ]:

"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s%(name)s:%(message)s')
file_handler = logging.FileHandler('/home/sauravpathak/data/models.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
"""

# In[ ]:


class dilated_residual_block(nn.Module):
    
    def __init__(self, dilation, input_length, samples_of_interest_indices, padded_target_field_length, condition_input_length, config):
        super().__init__()
        self.dilation =  dilation
        self.input_length = input_length
        self.condition_input_length = condition_input_length
        self.samples_of_interest_indices = samples_of_interest_indices
        self.padded_target_field_length = padded_target_field_length
        self.config = config
        self.conv1 = nn.Conv1d(self.config['model']['filters']['depths']['res'], 2*self.config['model']['filters']['depths']['res'],
                               kernel_size = self.config['model']['filters']['lengths']['res'], stride=1, bias = False, dilation = self.dilation,
                               padding=int(self.dilation))
        self.conv2 = nn.Conv1d(self.config['model']['filters']['depths']['res'], 
                               self.config['model']['filters']['depths']['res'] + self.config['model']['filters']['depths']['skip'],
                               1, stride=1, bias = False, padding=0)
        
        self.fc1 = nn.Linear(self.condition_input_length, 2*self.config['model']['filters']['depths']['res'], bias = False)
        
    
    def forward(self, data_x, condition_x):

        original_x = data_x

        # Data sub-block
        data_out = self.conv1(data_x)

        data_out_1 = layers.slicing(data_out, slice(0, self.config['model']['filters']['depths']['res'],1), 1) 

        data_out_2 = layers.slicing(data_out, slice(self.config['model']['filters']['depths']['res'], 
                                                    2*self.config['model']['filters']['depths']['res'],1), 1)
        
        # Condition sub-block
        condition_out = self.fc1(condition_x)
        condition_out = layers.expand_dims(condition_out, -1)

        condition_out_1 = layers.slicing(condition_out, slice(0, self.config['model']['filters']['depths']['res'],1), 1)

        condition_out_2 = layers.slicing(condition_out, slice(self.config['model']['filters']['depths']['res'], 
                                                    2*self.config['model']['filters']['depths']['res'],1), 1)
        
        condition_out_1 = condition_out_1.repeat(1,1,self.input_length)
        condition_out_2 = condition_out_2.repeat(1,1,self.input_length)

        data_out_1 = data_out_1 + condition_out_1
        
        data_out_2 = data_out_2 + condition_out_2

        tanh_out = torch.tanh(data_out_1)
        sigm_out = torch.sigmoid(data_out_2)

        data_x = tanh_out*sigm_out

        data_x = self.conv2(data_x)

        res_x = layers.slicing(data_x, slice(0, self.config['model']['filters']['depths']['res'],1), 1)
        
        skip_x = layers.slicing(data_x, slice(self.config['model']['filters']['depths']['res'], 
                                self.config['model']['filters']['depths']['res']+self.config['model']['filters']['depths']['skip'],1), 1)
        skip_x = layers.slicing(skip_x, slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1), 2)


        res_x = res_x + original_x

        return res_x, skip_x


# In[ ]:


# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Models.py

#Speech Denoising Wavenet Model

class DenoisingWavenet(nn.Module):

    def __init__(self, config, input_length=None, target_field_length=None):
        super().__init__()

        self.config = config
        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']

        self.num_condition_classes = config['dataset']['num_condition_classes']

        self.condition_input_length = self.get_condition_input_length(self.config['model']['condition_encoding'])
        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)

        if input_length is not None:
            self.input_length = int(input_length)
            self.target_field_length = int(self.input_length - (self.receptive_field_length - 1))
        if target_field_length is not None:
            self.target_field_length = int(target_field_length)
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))
        else:
            self.target_field_length = int(config['model']['target_field_length'])
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))

        self.target_padding = config['model']['target_padding']
        self.padded_target_field_length = self.target_field_length + 2 * self.target_padding
        self.half_target_field_length = int(self.target_field_length / 2)
        self.half_receptive_field_length = int(self.receptive_field_length / 2)
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        self.samples_of_interest_indices = self.get_padded_target_field_indices()
        self.target_sample_indices = self.get_target_field_indices()

        self.config['model']['num_residual_blocks'] = self.num_residual_blocks
        self.config['model']['receptive_field_length'] = self.receptive_field_length
        self.config['model']['input_length'] = self.input_length
        self.config['model']['target_field_length'] = self.target_field_length
        
        # Layers in the model
        self.conv1 = nn.Conv1d(1, self.config['model']['filters']['depths']['res'], self.config['model']['filters']['lengths']['res'],
                               stride=1, bias = False, padding=1)
        
        self.conv2 = nn.Conv1d(self.config['model']['filters']['depths']['res'], self.config['model']['filters']['depths']['final'][0],
                               self.config['model']['filters']['lengths']['final'][0], stride=1, bias = False,
                               padding=1)
        
        self.conv3 = nn.Conv1d(self.config['model']['filters']['depths']['final'][0], self.config['model']['filters']['depths']['final'][1],
                               self.config['model']['filters']['lengths']['final'][1], stride=1, bias = False,
                               padding=1)
        
        self.conv4 = nn.Conv1d(self.config['model']['filters']['depths']['final'][1], 1, 1, stride=1, bias = False, padding=0)
        
        self.fc1 = nn.Linear(self.condition_input_length, self.config['model']['filters']['depths']['res'], bias = False)
        
        self.fc2 = nn.Linear(self.condition_input_length, self.config['model']['filters']['depths']['final'][0], bias = False)
        self.fc3 = nn.Linear(self.condition_input_length, self.config['model']['filters']['depths']['final'][1], bias = False)
        
        self.dilated_layers = nn.ModuleList([dilated_residual_block(dilation, self.input_length, self.samples_of_interest_indices,
                                            self.padded_target_field_length, self.condition_input_length, self.config) for dilation in self.dilations])


    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length,
                     target_sample_index + self.half_target_field_length + 1)

    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length - self.target_padding,
                     target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_target_sample_index(self):
        return int(np.floor(self.input_length / 2.0))

    def get_condition_input_length(self, representation):

        if representation == 'binary':
            return int(np.ceil(np.log2(self.num_condition_classes)))
        else:
            return self.num_condition_classes

    def forward(self, x):

        data_input = x['data_input']

        condition_input = x['condition_input']

        data_expanded = layers.expand_dims(data_input, 1)
        
        data_input_target_field_length = layers.slicing(data_expanded,
            slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),2)

        data_out = self.conv1(data_expanded)

        condition_out = self.fc1(condition_input)
        condition_out = layers.expand_dims(condition_out, -1)
        condition_out = condition_out.repeat(1,1,self.input_length)
        
        data_out = data_out + condition_out

        skip_connections = []
        for _ in range(self.num_stacks):
            for layer in self.dilated_layers:
                data_out, skip_out = layer(data_out, condition_input)
                if skip_out is not None:
                    skip_connections.append(skip_out)

        data_out = torch.stack( skip_connections, dim = 0).sum(dim = 0)
        data_out = F.relu(data_out)

        data_out = self.conv2(data_out)

        condition_out = self.fc2(condition_input)
        condition_out = layers.expand_dims(condition_out, -1)
        condition_out = condition_out.repeat(1,1,self.padded_target_field_length)

        data_out = data_out + condition_out

        data_out = F.relu(data_out)
        
        data_out = self.conv3(data_out)
        
        condition_out = self.fc3(condition_input)
        condition_out = layers.expand_dims(condition_out, -1)
        condition_out = condition_out.repeat(1,1,self.padded_target_field_length)

        data_out = data_out + condition_out

        data_out = self.conv4(data_out)

        data_out_speech = data_out
        data_out_noise = data_input_target_field_length - data_out_speech

        data_out_speech = data_out_speech.squeeze_(1)

        data_out_noise = data_out_noise.squeeze_(1)

        return data_out_speech, data_out_noise


# In[ ]:


class TrainingConfig():
    def __init__(self, model, dataloader, config):
        self.config = config
        self.device = self.cuda_device()
        self.model = model.to(self.device)
        self.optimizer = self.get_optimizer()
        self.out_1_loss = self.get_out_1_loss()
        self.out_2_loss = self.get_out_2_loss()
        self.metric_fn = self.get_metrics_fn()
        self.num_epochs = self.config['training']['num_epochs']
        self.last_epoch = 0
        self.checkpoints_path = ''
        self.history_path = ''
        self.dataloader = dataloader
        self.scheduler = self.get_scheduler()
        self.train_losses = []
        self.train_metric = []
        self.valid_losses = []
        self.valid_metric = []
        
    def cuda_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda:1')
        else:
            return torch.device('cpu')

    def train(self, train_epoch_per_iter, valid_epoch_per_iter):
        
        # Early stopping intialization
        epochs_no_improve = 0
        valid_loss_min = np.Inf
        
        print('Training Started....')
        for epoch in tqdm(range(1,self.num_epochs+1-self.last_epoch)):
            counter = 0
            epoch = epoch + self.last_epoch
            self.model.train()
            batch_losses=[]
            batch_metric=[]
            for i, data in enumerate(self.dataloader['train_loader']):
                print(counter, end='\r')
                x, y = data
                self.optimizer.zero_grad()
                x = dict(map(lambda i: (i[0], i[1].to(self.device, dtype=torch.float32)), x.items()))
                y = dict(map(lambda i: (i[0], i[1].to(self.device, dtype=torch.float32)), y.items()))
                y_hat = self.model(x)
                loss = self.get_loss_fn(y_hat, y)
                batch_metric.append(self.metric_fn(y['data_output_1'].detach(), y_hat[0].detach()))  
                loss.backward()
                batch_losses.append(loss)
                self.optimizer.step()
                if counter >= train_epoch_per_iter:
                    break
                counter += 1
            self.train_losses.append(torch.stack(batch_losses, dim = 0).mean(dim = 0).detach().cpu().numpy())
            self.train_metric.append(torch.stack(batch_metric, dim = 0).mean(dim = 0).detach().cpu().numpy())
            print(f'Epoch - {epoch} Train-Loss : {self.train_losses[-1]} Train-mean-error : {self.train_metric[-1]}')
            counter = 0
            with torch.no_grad():
                self.model.eval()
                batch_losses=[]
                batch_metric=[]
                for i, data in enumerate(self.dataloader['valid_loader']):
                    x, y = data
                    x = dict(map(lambda i: (i[0], i[1].to(self.device, dtype=torch.float32)), x.items()))
                    y = dict(map(lambda i: (i[0], i[1].to(self.device, dtype=torch.float32)), y.items()))
                    y_hat = self.model(x)
                    loss = self.get_loss_fn(y_hat, y)
                    batch_metric.append(self.metric_fn(y['data_output_1'].detach(), y_hat[0].detach()))       
                    batch_losses.append(loss)
                    if counter >= valid_epoch_per_iter:
                        break
                    counter += 1
                self.valid_losses.append(torch.stack(batch_losses, dim = 0).mean(dim = 0).detach().cpu().numpy())
                self.valid_metric.append(torch.stack(batch_metric, dim = 0).mean(dim = 0).detach().cpu().numpy())
                valid_loss = np.mean(self.valid_losses)
            self.scheduler.step(self.valid_metric[-1])
            print(f'Epoch - {epoch} Valid-Loss : {self.valid_losses[-1]} Valid-mean-error : {self.valid_metric[-1]}')
            if valid_loss < valid_loss_min:
                # Save model
                state = {'model_state':self.model.state_dict(), 'optimizer_state':self.optimizer.state_dict()}
                torch.save(state, 'data/NSDTSEA/checkpoints/config1_epoch{:04d}.pth'.format(epoch))
                if epoch != 1:
                    checkpoints = os.listdir(self.checkpoints_path)
                    checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
                    last_checkpoint = checkpoints[-2]
                    last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
                    os.remove(last_checkpoint_path)
                history = list(zip(self.train_losses, self.valid_losses, self.train_metric, self.valid_metric))
                history = pd.DataFrame(history,columns=['train_losses', 'valid_losses', 'train_metric','valid_metric'])
                history.to_pickle(self.history_path)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                best_epoch = epoch
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= self.config['training']['early_stopping_patience']:
                    print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f}')
                    break

    def get_metrics_fn(self):

        return lambda y_true, y_pred: F.l1_loss(y_true[:,1:-2], y_pred[:,1:-2])

    
    def setup_model(self, load_checkpoint=None, print_model_summary=False):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')
        self.history_path = os.path.join(self.config['training']['path'], 'history', 'history.pkl')

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            if load_checkpoint is not None:
                last_checkpoint_path = load_checkpoint
                self.last_epoch = int(last_checkpoint_path[38:42])
            else:
                checkpoints = os.listdir(self.checkpoints_path)
                checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
                last_checkpoint = checkpoints[-1]
                last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
                self.last_epoch = int(last_checkpoint[13:17])
                print('Loading model from epoch: %d' % self.last_epoch)
            state = torch.load(last_checkpoint_path)
            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            history = pd.read_pickle(self.history_path)
            self.train_losses = list((history['train_losses']))
            self.valid_losses = list(history['valid_losses'])
            self.train_metric = list(history['train_metric'])
            self.valid_metric = list(history['valid_metric'])

        else:
            print('Training From Scratch....')

            if not os.path.exists(self.config['training']['path']):
                os.mkdir(self.config['training']['path'])

            if not os.path.exists(self.checkpoints_path):
                os.mkdir(self.checkpoints_path)

            if not os.path.exists(os.path.join(self.config['training']['path'], 'history')):
                os.mkdir(os.path.join(self.config['training']['path'], 'history'))

            self.last_epoch = 0

        #if print_model_summary:
            #print(summary(self.model, (1,48000)))

    def get_optimizer(self):
        
        return optim.Adam(self.model.parameters(), lr=self.config['optimizer']['lr'], weight_decay=self.config['optimizer']['decay'])

    def get_scheduler(self):
        
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.config['training']['early_stopping_patience']/2,
                                                    cooldown=self.config['training']['early_stopping_patience']/4, verbose = True)
    
    def get_out_1_loss(self):

        if self.config['training']['loss']['out_1']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_1']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_1']['l1'],
            self.config['training']['loss']['out_1']['l2'])

    def get_out_2_loss(self):

        if self.config['training']['loss']['out_2']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_2']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_2']['l1'],
            self.config['training']['loss']['out_2']['l2'])
    
    def get_loss_fn(self, y_hat, y):
        target_speech = y['data_output_1']
        target_noise = y['data_output_2']
        output_speech = y_hat[0]
        output_noise = y_hat[1]
        loss1 = self.out_1_loss(target_speech, output_speech)
        loss2 = self.out_2_loss(target_noise, output_noise)
        loss = loss1 + loss2
        
        return loss

class PredictConfig():
    def __init__(self, model, checkpoint_path):
        self.device = self.cuda_device()
        self.model = model.to(self.device)
        self.checkpoint_path = checkpoint_path
        
    def cuda_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def get_trained_model(self):
        state = torch.load(self.checkpoint_path)
        self.model.load_state_dict(state['model_state'])
        return self.model

    def denoise_batch(self, inputs):
        with torch.no_grad():
            self.model.eval() 
            inputs = dict(map(lambda i: (i[0], i[1].to(self.device, dtype=torch.float32)), inputs.items()))
            y_hat = self.model(inputs)
        return y_hat
