import torch
import blocks
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from collections import OrderedDict
from adain import adaptive_instance_norm
import tensorly as tl
from tensorly.decomposition import tucker

tl.set_backend('pytorch')

class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, 
                 use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, 
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x



class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) # norm to [0,1] NxHxWx1
        hg, wg = hg*2-1, wg*2-1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)



class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):


        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=bn)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)

        return output


class FusionNetwork(nn.Module):
    def __init__(self, size=256):
        super(FusionNetwork, self).__init__()
        self.size = size
        self.extractor = blocks.Vgg19()
        self.fu1 = blocks.FusionBlock1(64, 8)
        self.fu2 = blocks.FusionBlock2(128, 16)
        self.fu3 = blocks.FusionBlock3(256, 32)
        self.fu4 = blocks.FusionBlock4(512, 32)

        self.final_conv1 = nn.Conv2d(32, 12, 1, stride=1, padding=0, bias=False)
        self.final_conv2 = nn.Conv2d(512, 12, 1, stride=1, padding=0, bias=False)

    def forward(self, content, style):
        lr_c = F.interpolate(content, (self.size, self.size),
                            mode='bilinear', align_corners=True)
        lr_s = F.interpolate(style, (self.size, self.size),
                            mode='bilinear', align_corners=True)
        feats_c = self.extractor(lr_c)
        feats_s = self.extractor(lr_s)
        
        
        feat_adain1 = adaptive_instance_norm(feats_c[1], feats_s[1])
        out_c1, out_s1 = self.fu1(feats_c[0], feats_s[0], feat_adain1)
        
        
        
        feat_adain2 = adaptive_instance_norm(feats_c[2], feats_s[2])
        out_c2, out_s2 = self.fu2(out_c1, out_s1, feat_adain2)
        feat_adain3 = adaptive_instance_norm(feats_c[3], feats_s[3])
        out_c3, out_s3 = self.fu3(out_c2, out_s2, feat_adain3)
        feat_adain4 = adaptive_instance_norm(feats_c[3], feats_s[3])
        out_c4, out_s4 = self.fu4(out_c3, out_s3, feat_adain4)
        
        out_c4 = self.final_conv1(out_c4)
        
        out_s4 = self.final_conv2(out_s4)
        
        
        c_grid = out_c4.view(-1, 12, 4, 16, 16) 
        s_grid = out_s4.view(-1, 12, 4, 16, 16) 
        
        
        
        c_grid = tl.tensor(c_grid)
        core, factors = tucker(c_grid, rank=[12, 2, 8, 8])
        c_grid = tl.tucker_to_tensor((core, factors))
        
        
        
        s_grid = tl.tensor(s_grid)
        core, factors = tucker(s_grid, rank=[12, 2, 8, 8])
        s_grid = tl.tucker_to_tensor((core, factors))
        
        
        return c_grid, s_grid




class BilateralNetwork(nn.Module):
    def __init__(self, size=256):
        super(BilateralNetwork, self).__init__()
        self.size = size
        self.Fnet = FusionNetwork(size)
        self.guide = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        
        self.final = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 3, 3, stride=1, padding=1))

    def forward(self, content, style):
        c_coeff, s_coeff = self.Fnet(content, style)
        guide = self.guide(content)
        c_slice_coeffs = self.slice(c_coeff, guide)
        s_slice_coeffs = self.slice(s_coeff, guide)
        c_output = self.apply_coeffs(c_slice_coeffs, content)
        s_output = self.apply_coeffs(s_slice_coeffs, style)
        
        output = adaptive_instance_norm(c_output, s_output)
        
        output = self.final(output)      
        output = torch.sigmoid(output)
        return output




class Vgg19(torch.nn.Module):
    def __init__(self, weight_path='vgg19.pth'):
        super(Vgg19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load(weight_path))
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg19.features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg19.features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg19.features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg19.features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        conv1_1 = self.slice1(inputs)
        conv2_1 = self.slice2(conv1_1)
        conv3_1 = self.slice3(conv2_1)
        conv4_1 = self.slice4(conv3_1)
        
        return conv1_1, conv2_1, conv3_1, conv4_1
        


network = BilateralNetwork()

c = torch.zeros(1, 3, 3840, 2160)
s = torch.zeros(1, 3, 3840, 2160)

output = network(c, s)

print(output.shape)