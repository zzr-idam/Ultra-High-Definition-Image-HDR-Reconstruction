import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from adain import adaptive_instance_norm



class FusionBlock1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionBlock1, self).__init__()
        self.conv_in = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
                nn.Conv2d(ch_out, ch_out, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_out1 = nn.Sequential(
                nn.Conv2d(ch_out, 128, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_fu = nn.Sequential(
                nn.Conv2d(128, 128, 1, stride=1),
                nn.ReLU(inplace=True))
        
        
        
    def forward(self, feat_s, feat_c, feat_adain):
        feat_s = self.conv_in(feat_s)
        feat_c = self.conv_in(feat_c)
        

        
        feat_c = self.conv_out(feat_c)
        feat_s = self.conv_out1(feat_s)
        #print(feat_c.size(), feat_s.size())
        #feat_norm = adaptive_instance_norm(feat_c, feat_s)
        #print(feat_norm.size(), feat_adain.size())
        

        
        output = feat_s + feat_adain
        output = self.conv_fu(output)
        return output, feat_c
        
class FusionBlock2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionBlock2, self).__init__()
        self.conv_in = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_in_c = nn.Sequential(
                nn.Conv2d(8, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
                nn.Conv2d(ch_out, ch_out, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_out1 = nn.Sequential(
                nn.Conv2d(ch_out, 256, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_fu = nn.Sequential(
                nn.Conv2d(256, 256, 1, stride=1),
                nn.ReLU(inplace=True))
        
        
        
    def forward(self, feat_s, feat_c, feat_adain):
        feat_s = self.conv_in(feat_s)
        feat_c = self.conv_in_c(feat_c)
        

        
        feat_c = self.conv_out(feat_c)
        feat_s = self.conv_out1(feat_s)
        #print(feat_c.size(), feat_s.size())
        #feat_norm = adaptive_instance_norm(feat_c, feat_s)
        #print(feat_norm.size(), feat_adain.size())
        

        
        output = feat_s + feat_adain
        output = self.conv_fu(output)
        return output, feat_c

class FusionBlock3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionBlock3, self).__init__()
        self.conv_in = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_in_c = nn.Sequential(
                nn.Conv2d(16, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
                nn.Conv2d(ch_out, ch_out, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_out1 = nn.Sequential(
                nn.Conv2d(ch_out, 512, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_fu = nn.Sequential(
                nn.Conv2d(512, 512, 1, stride=1),
                nn.ReLU(inplace=True))
        
        
        
    def forward(self, feat_s, feat_c, feat_adain):
        feat_s = self.conv_in(feat_s)
        feat_c = self.conv_in_c(feat_c)
        

        
        feat_c = self.conv_out(feat_c)
        feat_s = self.conv_out1(feat_s)
        #print(feat_c.size(), feat_s.size())
        #feat_norm = adaptive_instance_norm(feat_c, feat_s)
        #print(feat_norm.size(), feat_adain.size())

        
        output = feat_s + feat_adain
        output = self.conv_fu(output)
        return output, feat_c
    
class FusionBlock4(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionBlock4, self).__init__()
        self.conv_in = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_in_c = nn.Sequential(
                nn.Conv2d(32, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
                nn.Conv2d(ch_out, ch_out, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_out1 = nn.Sequential(
                nn.Conv2d(ch_out, 512, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_fu = nn.Sequential(
                nn.Conv2d(512, 32, 1, stride=1),
                nn.ReLU(inplace=True))
        
        
        
    def forward(self, feat_s, feat_c, feat_adain):
        feat_s = self.conv_in(feat_s)
        feat_c = self.conv_in_c(feat_c)
        

        
        feat_c = self.conv_out(feat_c)
        feat_s = self.conv_out1(feat_s)
        #print(feat_c.size(), feat_s.size())
        #feat_norm = adaptive_instance_norm(feat_c, feat_s)
        #print(feat_norm.size(), feat_adain.size())
               
        feat_s = F.interpolate(feat_s, (32, 32),
                            mode='bilinear', align_corners=True)
        
        feat_c = F.interpolate(feat_s, (32, 32),
                            mode='bilinear', align_corners=True)
        
        output = feat_s + feat_adain

        output = self.conv_fu(output)
        return output, feat_c

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
        

if __name__ == '__main__':
    model = nn.Conv2d(3, 3, 3, stride=2, padding=1)
    test = torch.ones(1, 3, 256, 256)
    output = model(test)
    print(output.size())
    
    
        
            

        
        
    