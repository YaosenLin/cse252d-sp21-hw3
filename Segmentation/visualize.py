import torch
from torch.autograd import Variable
import dataLoader
import argparse
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io
import matplotlib.pyplot as plt

colormap = io.loadmat('colormap.mat')['cmap']

# Initialize image batch
imageBatch = Variable(torch.FloatTensor(1, 3, 300, 300))
labelBatch = Variable(torch.FloatTensor(1, 21, 300, 300))
maskBatch = Variable(torch.FloatTensor(1, 1, 300, 300))
labelIndexBatch = Variable(torch.LongTensor(1, 1, 300, 300))

encoder_unet = model.encoderDilation()
decoder_unet = model.decoderDilation()
encoder_dilation = model.encoder()
decoder_dilation = model.decoder()
encoder_spp = model.encoderSPP()
decoder_spp = model.decoderSPP()


encoder_unet.load_state_dict(torch.load('unet-2/encoder_119.pth'))
decoder_unet.load_state_dict(torch.load('unet-2/decoder_119.pth'))
encoder_unet = encoder_unet.eval()
decoder_unet = decoder_unet.eval()

encoder_dilation.load_state_dict(torch.load('dilation-3/encoder_119.pth'))
decoder_dilation.load_state_dict(torch.load('dilation-3/decoder_119.pth'))
encoder_dilation = encoder_dilation.eval()
decoder_dilation = decoder_dilation.eval()

encoder_spp.load_state_dict(torch.load('spp-3/encoder_119.pth'))
decoder_spp.load_state_dict(torch.load('spp-3/decoder_119.pth'))
encoder_spp = encoder_spp.eval()
decoder_spp = decoder_spp.eval()

# Move network and containers to gpu
device = 'cuda'

imageBatch = imageBatch.to(device)
labelBatch = labelBatch.to(device)
labelIndexBatch = labelIndexBatch.to(device)
maskBatch = maskBatch.to(device)

encoder_unet = encoder_unet.to(device)
decoder_unet = decoder_unet.to(device)
encoder_dilation = encoder_dilation.to(device)
decoder_dilation = decoder_dilation.to(device)
encoder_spp = encoder_spp.to(device)
decoder_spp = decoder_spp.to(device)


# Initialize dataLoader
dataset = dataLoader.BatchLoader(
        imageRoot = '/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/JPEGImages',
        labelRoot = '/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/SegmentationClass',
        fileList = '/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
        )
dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

for i, dataBatch in enumerate(dataloader):

    # Read data
    imageBatch = Variable(dataBatch['image']).to(device)
    labelBatch = Variable(dataBatch['label']).to(device)
    labelIndexBatch = Variable(dataBatch['labelIndex']).to(device)
    maskBatch = Variable(dataBatch['mask']).to(device)

    # Test network
    x1, x2, x3, x4, x5 = encoder_unet(imageBatch)
    p_unet = decoder_unet(imageBatch, x1, x2, x3, x4, x5)

    x1, x2, x3, x4, x5 = encoder_dilation(imageBatch)
    p_dilation = decoder_dilation(imageBatch, x1, x2, x3, x4, x5)

    x1, x2, x3, x4, x5 = encoder_spp(imageBatch)
    p_spp = decoder_spp(imageBatch, x1, x2, x3, x4, x5)


    vutils.save_image(imageBatch.data , 'plots/image_%d.png' % i, padding=0, normalize = True)
    gt = utils.save_label(labelBatch.data, maskBatch.data, colormap, 'plots/labelGT_%d.png' % i, nrows=1, ncols=1 )
    p_unet = utils.save_label(-p_unet.data, maskBatch.data, colormap, 'plots/label_unet_%d.png' % i, nrows=1, ncols=1 )
    p_dilation = utils.save_label(-p_dilation.data, maskBatch.data, colormap, 'plots/label_dilation_%d.png' % i, nrows=1, ncols=1 )
    p_spp = utils.save_label(-p_spp.data, maskBatch.data, colormap, 'plots/label_spp_%d.png' % i, nrows=1, ncols=1 )

    print(type(gt))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(gt)
    plt.xticks([])
    plt.yticks([])
    plt.title('ground truth')
    plt.subplot(2, 2, 2)
    plt.imshow(p_unet)
    plt.xticks([])
    plt.yticks([])
    plt.title('unet')
    plt.subplot(2, 2, 3)
    plt.imshow(p_dilation)
    plt.xticks([])
    plt.yticks([])
    plt.title('+dilation')
    plt.subplot(2, 2, 4)
    plt.imshow(p_spp)
    plt.xticks([])
    plt.yticks([])
    plt.title('+dilation+spp')
    plt.savefig('image_%d.png'%i)

    if i == 3:
        break


# import matplotlib.pyplot as plt

# for i in range(3):
#     plt.figure()
#     plt.subplot(2, 2, 1)
#     plt.imshow(plt.imread('plots/image_%d.png' % i))
#     plt.subplot(2, 2, 2)
#     plt.imshow()

