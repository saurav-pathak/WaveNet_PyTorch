# WaveNet-PyTorch
 Speech Denoising WaveNet Implmentation in PyTorch
# Contents
1. ./main.py or main1.py: python script for training or inference
2. ./data/wavenet/models.py: python script containing speech denoising model
3. ./data/wavenet/denoise.py: python script for speech denoising during inference
4. ./data/wavenet/layers.py or util.py: additional support modules
5. ./data/wavenet/dataset.py: python script for dataset preparation
6. ./data/wavenet/config.py: model parameters
7. ./speech_denoise_test.ipynb: Python notebook for testing model
# Dataset
The "Noisy speech database for training speech enhancement algorithms and TTS models" (NSDTSEA) is used for training the model. It is provided by the University of Edinburgh, School of Informatics, Centre for Speech Technology Research (CSTR).
Please download from link: https://datashare.is.ed.ac.uk/handle/10283/1942 and save to ./data/NSDTSEA
# Original Paper
A Wavenet for Speech Denoising
link: https://arxiv.org/abs/1706.07162
# Additional files
./data/NSDTSEA/checkpoints contains a pretrained model which can be directly used for inference. 
