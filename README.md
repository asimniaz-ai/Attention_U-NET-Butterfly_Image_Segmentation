# Attention_U-NET-Butterfly_Image_Segmentation
This model incorporates self-attention gating modules to the U-Net architecture. This repository contains trained weights for segmenting Butterfly images dataset of Learning Models for Object Recognition from Natural Language Descriptions.


This repository contains code for the Butterfly images Segmentation using Attention U-Net with PyTorch on the LEADS BUTTERFLY dataset.

#### NVIDIA 3090
#### CUDA 11.3
#### Python 3.7
#### PyTorch (conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia)
#### [LEADS BUTTERFLY](http://www.josiahwang.com/dataset/leedsbutterfly/)

This dataset contains images and textual descriptions for ten categories (species) of butterflies. The image dataset comprises 832 images in total, with the distribution ranging from 55 to 100 images per category.

#### Download the dataset and paste in the same folder as your code is in. Divide the dataset in two splits: train and test; Update the data paths in train and test files accordingly.

To use multiple GPUs, you could use the following in train.py
device = torch.device('cuda')
model = build_unet()
model = nn.DataParallel(model) #optional; could be commented
model = model.to(device)
