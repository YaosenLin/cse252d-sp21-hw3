import argparse
import dataLoader
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import os
import numpy as np
import scipy.io as io

import utils
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--imageRoot', default='/datasets/cs252-sp21-A00-public/hw3_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cs252-sp21-A00-public/hw3_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cs252-sp21-A00-public/hw3_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', help='path to input images' )
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')
parser.add_argument('--experiment', default='checkpoint', help='the path to store sampled images and models')
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not' )
parser.add_argument('--nClasses', type=int, default=21, help='the number of classes' )
parser.add_argument('--batchSize', type=int, default=16, help='the size of a batch')
parser.add_argument('--nepoch', type=int, default=200, help='the training epoch')
parser.add_argument('--initialLR', type=float, default=0.0001, help='the initial learning rate')
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )

# The detail network setting
opt = parser.parse_args()
# print(opt)

if __name__ == "__main__":
    # Save all the codes
    if not os.path.exists(opt.experiment):
        os.system('mkdir %s' % opt.experiment)
        os.system('cp *.py %s' % opt.experiment)
    else:
        raise Exception('Clean "checkpoint" first')

    if torch.cuda.is_available() and opt.noCuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# Uncomment to train a certain encoder/decoder
#     encoder = encoder()
#     decoder = decoder()
    encoder = encoderDilation()
    decoder = decoderDilation()
#     encoder = encoderSPP()
#     decoder = decoderSPP()
    loadPretrainedWeight(encoder)
    encoder.train()
    decoder.train()
    

    # Initialize image batch
    imageBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 300, 300))
    labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.nClasses, 300, 300))
    maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, 300, 300))
    labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, 300, 300))
    colormap = io.loadmat(opt.colormap)['cmap']

    # Move network and containers to gpu
    if not opt.noCuda:
        device = 'cuda'
    else:
        device = 'cpu'

    imageBatch = imageBatch.to(device)
    labelBatch = labelBatch.to(device)
    labelIndexBatch = labelIndexBatch.to(device)
    maskBatch = maskBatch.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=opt.initialLR, weight_decay=5e-4)

    dataset = dataLoader.BatchLoader(
            imageRoot = opt.imageRoot,
            labelRoot = opt.labelRoot,
            fileList = opt.fileList,
            imHeight = 300,
            imWidth = 300
        )
    
    loader = DataLoader(dataset, batch_size=opt.batchSize, num_workers=0, shuffle=True)

    loss_record = []
    accuracy = np.zeros((opt.nClasses), dtype=np.float32)
    confusion = np.zeros((opt.nClasses, opt.nClasses), dtype=np.int64)

    iteration = 0
    for epoch in range(opt.nepoch):
        confusion = np.zeros((opt.nClasses, opt.nClasses), dtype=np.int64)
        log = open('{0}/training_log_{1}.txt'.format(opt.experiment, epoch), 'w')
        np.random.seed()
        if ((epoch+1) % 20 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            np.save('%s/loss.npy' % opt.experiment, np.array(loss_record))
            np.save('%s/confusion.npy' % opt.experiment, np.array(confusion))
            torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch))
            torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch))

        for i, dataBatch in enumerate(loader):
            iteration += 1

            # Read data
            imageBatch = Variable(dataBatch['image']).to(device)
            labelBatch = Variable(dataBatch['label']).to(device)
            labelIndexBatch = Variable(dataBatch['labelIndex']).to(device).long()
            maskBatch = Variable(dataBatch['mask']).to(device)

            optimizer.zero_grad()

            x1, x2, x3, x4, x5 = encoder(imageBatch)
            prediction = decoder(imageBatch, x1, x2, x3, x4, x5)
            loss = torch.mean( prediction * labelBatch )

            loss.backward()

            optimizer.step()
            
            hist = utils.computeAccuracy(prediction, labelIndexBatch, maskBatch)
            confusion += hist

            loss_record.append(loss.cpu().data.item())

            print('Epoch %d Iteration %d: Loss %.5f, Mean Loss %.5f' % (epoch, iteration, loss_record[-1], np.mean(np.array(loss_record[:]))))
            
            log.write('Epoch %d Iteration %d: Loss %.5f, Mean Loss %.5f' % (epoch, iteration, loss_record[-1], np.mean(np.array(loss_record[:]))))
            

        for n in range(0, opt.nClasses):
            rSum = np.sum(confusion[n, :])
            cSum = np.sum(confusion[:, n])
            correct = confusion[n, n]
            accuracy[n] = float(correct) / max(float(rSum + cSum - correct), 1e-5)
        print('Epoch %d Iteration %d: Mean Accuracy %.5f' % (epoch, iteration, np.mean(accuracy)))
        log.write('Epoch %d Iteration %d: Mean Accuracy %.5f' % (epoch, iteration, np.mean(accuracy)))
        
        # np.save('%s/loss.npy' % opt.experiment, np.array(loss_record))
        # np.save('%s/confusion.npy' % opt.experiment, np.array(confusion))
        # torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch))
        # torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch))

