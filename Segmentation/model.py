import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models.resnet as resnet


class ResBlock(nn.Module):
    def __init__(self, inch, outch, stride=1, dilation=1):
        # Residual Block
        # inch: input feature channel
        # outch: output feature channel
        # stride: the stride of convolution layer
        super(ResBlock, self).__init__()
        assert(stride == 1 or stride == 2)

        self.conv1 = nn.Conv2d(inch, outch, 3, stride, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm2d(outch)
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(outch)

        if inch != outch:
            self.mapping = nn.Sequential(
                nn.Conv2d(inch, outch, 1, stride, bias=False),
                nn.BatchNorm2d(outch)
            )
        else:
            self.mapping = None

    def forward(self, x):
        y = x
        if not self.mapping is None:
            y = self.mapping(y)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out += y
        out = F.relu(out, inplace=True)

        return out


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 2)
        self.b3_2 = ResBlock(256, 256, 1)

        self.b4_1 = ResBlock(256, 512, 2)
        self.b4_2 = ResBlock(512, 512, 1)

    def forward(self, im):
        x1 = F.relu(self.bn1(self.conv1(im)), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1)))
        x3 = self.b2_2(self.b2_1(x2))
        x4 = self.b3_2(self.b3_1(x3))
        x5 = self.b4_2(self.b4_1(x4))
        return x1, x2, x3, x4, x5


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.conv1 = nn.Conv2d(512+256+128, 512, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(21)
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(21)
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.sf = nn.Softmax(dim=1)

    def forward(self, im, x1, x2, x3, x4, x5):

        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear')
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear')
        y1 = F.relu(
            self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.sf(self.conv4(y3))

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        p = -torch.log(torch.clamp(y4, min=1e-8))
        return p


# 2. `UNet with dilation`: We modify the encoder to a dilated `resnet18` 
# as described in Section 2 of [1] 
# (You are not required to consider degridding in Section 4 of [1]). 
# We set the stride of the last 4 residual blocks to be 1 
# so that the highest level feature maps will be $4 \times 4$ times larger. 
# To increase the receptive field, we set the dilation of residual blocks 
# that are fourth and third from the end to be 2, 
# while the dilation of the residual blocks that are first and second 
# from the end are set to 4.  
# The decoder should be modified accordingly. 
# Implement your new encoder and decoder under class 
# `encoderDilation` and `decoderDilation`. 
# Ensure that for images of arbitrary shapes, 
# the decoder will give segmentation outputs of the same shape.
# **[15 points]**

class encoderDilation(nn.Module):
    def __init__(self):

        # IMPLEMENT YOUR CODE HERE

        super(encoderDilation, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, 2)
        self.b3_2 = ResBlock(256, 256, 1, 2)

        self.b4_1 = ResBlock(256, 512, 1, 4)
        self.b4_2 = ResBlock(512, 512, 1, 4)

    def forward(self, im):

        # IMPLEMENT YOUR CODE HERE

        x1 = F.relu(self.bn1(self.conv1(im)), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1)))
        x3 = self.b2_2(self.b2_1(x2))
        x4 = self.b3_2(self.b3_1(x3))
        x5 = self.b4_2(self.b4_1(x4))
        return x1, x2, x3, x4, x5


class decoderDilation(nn.Module):
    def __init__(self, isSpp=False):
        
        # IMPLEMENT YOUR CODE HERE

        super(decoderDilation, self).__init__()
        self.conv1 = nn.Conv2d(512+256+128, 512, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(21)
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(21)
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.sf = nn.Softmax(dim=1)

    def forward(self, im, x1, x2, x3, x4, x5):

        # IMPLEMENT YOUR CODE HERE

        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear')
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear')
        y1 = F.relu(self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.sf(self.conv4(y3))

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        p = -torch.log(torch.clamp(y4, min=1e-8))

        return p


class convolve_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn=False):
        super(convolve_bn, self).__init__()

        bias = not bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class SPP(nn.Module):
    def __init__(self, in_channels):

        super(SPP, self).__init__()
        size = [(1, 1), (2, 2), (3, 3), (6, 6)]

        self.pools = nn.ModuleList()
        for _, s in enumerate(size):
            pool_module = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=s),
                convolve_bn(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, bn=True)
            )
            self.pools.append(pool_module)

    def forward(self, x, nh, nw):
        
        out_list = [F.interpolate(x, [nh, nw], mode='bilinear')]
        
        for pool_module in self.pools:
            out = pool_module(x)
            out = F.interpolate(out, [nh, nw], mode='bilinear')
            out_list.append(out)

        output = torch.cat(out_list, dim=1)
        return output


class encoderSPP(nn.Module):
    def __init__(self):

        # IMPLEMENT YOUR CODE HERE

        super(encoderSPP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, 2)
        self.b3_2 = ResBlock(256, 256, 1, 2)

        self.b4_1 = ResBlock(256, 512, 1, 4)
        self.b4_2 = ResBlock(512, 512, 1, 4)

    def forward(self, im):

        # IMPLEMENT YOUR CODE HERE

        x1 = F.relu(self.bn1(self.conv1(im)), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1)))
        x3 = self.b2_2(self.b2_1(x2))
        x4 = self.b3_2(self.b3_1(x3))
        x5 = self.b4_2(self.b4_1(x4))
        return x1, x2, x3, x4, x5


class decoderSPP(nn.Module):
    def __init__(self):
        
        super(decoderSPP, self).__init__()
        self.spp = SPP(in_channels=512)

        self.conv1 = nn.Conv2d(1024 + 256 + 128, 512, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64 + 21, 21, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(21)
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(21)
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.sf = nn.Softmax(dim=1)

    def forward(self, im, x1, x2, x3, x4, x5):
        
        _, _, nh, nw = x3.size()
        x5 = self.spp(x5, nh, nw)

        x4 = F.interpolate(x4, [nh, nw], mode='bilinear')
        y1 = F.relu(self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.sf(self.conv4(y3))

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        p = -torch.log(torch.clamp(y4, min=1e-8))

        return p


def loadPretrainedWeight(network, isOutput=False):
    paramList = []
    resnet18 = resnet.resnet18(pretrained=True)
    for param in resnet18.parameters():
        paramList.append(param)

    cnt = 0
    for param in network.parameters():
        if paramList[cnt].size() == param.size():
            param.data.copy_(paramList[cnt].data)
            #param.data.copy_(param.data )
            if isOutput:
                print(param.size())
        else:
            print(param.shape, paramList[cnt].shape)
            break
        cnt += 1
