import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CoAtNet import CoAtNet_Seg
from .MobileVit import MobileViT_Seg
from .convit import Convit_seg
from .UNetFormer import UNetFormer
from .EfficientViT import EfficientViT
from .DCSwin import DCSwin


# import segmentation_models_pytorch as smp
# from seg_hrnet import HighResolutionNet
# from deeplabv3plus import DeepLabV3Plus
# from refinenet import rf101

import utils

class FCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(FCN,self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.last =  nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x

class Skip_FCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(Skip_FCN,self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.last =  nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,inputs):
        x = F.relu(self.conv1(inputs))
        x1 = F.relu(self.conv2(x))
        x2 = self.conv3(x1)
        x2 = F.relu(x + x2)
        x3 = F.relu(self.conv4(x2))
        x4 = self.conv5(x3)
        x4 = F.relu(x2 + x4)
        x5 = self.last(x4)
        return x5

class HRModule(nn.Module):
    def __init__(self, input_chs):
        super(HRModule,self).__init__()

        self.bn_momentum = 0.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_chs // 4, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(input_chs // 32, momentum=self.bn_momentum),
            nn.ReLU()
        )
        #self.gap = nn.AdaptiveAvgPool2d(1)
        #self.conv4 = nn.Sequential(
       #     nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
      #      nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
     #       nn.ReLU()
    #    )
      #  self.conv5 = nn.Sequential(
        #    nn.Conv2d(input_chs, input_chs // 4, kernel_size=1, stride=1, padding=0),
       #     nn.BatchNorm2d(input_chs // 4, momentum=self.bn_momentum),
      #      nn.ReLU()
     #   )
      #  self.conv6 = nn.Sequential(
       #     nn.Conv2d(input_chs, input_chs // 64, kernel_size=1, stride=1, padding=0),
        #    nn.BatchNorm2d(input_chs // 64, momentum=self.bn_momentum),
         #   nn.ReLU()
       # )
        self.f_conv = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(input_chs + input_chs // 4+ input_chs // 32, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
      #  f1 = self.conv4(self.gap(inputs))
       # f2 = self.conv5(self.gap(inputs))
      #  f3 = self.conv6(self.gap(inputs))
      #  x1 = torch.mul(x1, f1)
      #  x2 = torch.mul(x2, f2)
      #  x3 = torch.mul(x3, f3) #, x3
        outputs = self.f_conv(torch.cat([x1, x2,x3], dim=1)) + inputs

        return outputs

class L2HNet(nn.Module):
    def __init__(self, insize,input_chs, num_output_classes):
        super(L2HNet,self).__init__()
        c5=4
        c3=2
        self.startconv = nn.Conv2d(insize, input_chs, kernel_size=3, stride=1, padding=1)
        self.bn_momentum = 0.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_chs // c3, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(input_chs // c5, momentum=self.bn_momentum),
            nn.ReLU()
         )   
      #----------------------------------------------------------------
        self.conv4 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_chs // c3, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(input_chs // c5, momentum=self.bn_momentum),
            nn.ReLU()
         )   
      #----------------------------------------------------------------
        self.conv7 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_chs // c3, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(input_chs // c5, momentum=self.bn_momentum),
            nn.ReLU()
         )   
      #----------------------------------------------------------------
        self.conv10 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_chs // c3, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(input_chs // c5, momentum=self.bn_momentum),
            nn.ReLU()
         )   
      #----------------------------------------------------------------
        self.conv13 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_chs // c3, momentum=self.bn_momentum),
            nn.ReLU()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs // c5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(input_chs // c5, momentum=self.bn_momentum),
            nn.ReLU()
         )   
      #----------------------------------------------------------------
        
        self.f_conv1 = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(input_chs + input_chs // c3+ input_chs // c5, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
            )
        self.f_conv2 = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(input_chs + input_chs // c3+ input_chs // c5, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
            )
        self.f_conv3 = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(input_chs + input_chs // c3+ input_chs // c5, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
            )
        self.f_conv4 = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(input_chs + input_chs // c3+ input_chs // c5, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
            )  
        self.f_conv5 = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(input_chs + input_chs // c3+ input_chs // c5, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
            nn.ReLU()
            )            
   #     self.branch_conv = nn.Sequential( #+ input_chs // 64
  #          nn.Conv2d(input_chs*5, input_chs, kernel_size=1, stride=1, padding=0),
    #        nn.BatchNorm2d(input_chs, momentum=self.bn_momentum),
      #      nn.ReLU()
     #   )
        self.last =  nn.Conv2d(input_chs, num_output_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, inputs):
        inputs=F.relu(self.startconv(inputs))
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        outputs1 = self.f_conv1(torch.cat([x1, x2,x3], dim=1)) + inputs
        x4 = self.conv4(outputs1)
        x5 = self.conv5(outputs1)
        x6 = self.conv6(outputs1)
        outputs2 = self.f_conv2(torch.cat([x4, x5,x6], dim=1)) + outputs1
        # self.featuremap3 = torch.max(outputs2, dim=1, keepdim=True)
        # self.featuremap3=self.featuremap3[0].detach()
        # self.featuremap4 = torch.mean(outputs2, dim=1, keepdim=True).detach()
        x7 = self.conv7(outputs2)
        x8 = self.conv8(outputs2)
        x9 = self.conv9(outputs2)
        outputs3 = self.f_conv3(torch.cat([x7, x8,x9], dim=1)) + outputs2
        x10 = self.conv10(outputs3)
        x11 = self.conv11(outputs3)
        x12 = self.conv12(outputs3)
        outputs4 = self.f_conv4(torch.cat([x10, x11,x12], dim=1)) + outputs3 #output4
        x13 = self.conv13(outputs4)
        x14 = self.conv14(outputs4)
        x15 = self.conv15(outputs4)
        outputs5 = self.f_conv5(torch.cat([x13, x14,x15], dim=1)) + outputs4        
       # Finaloutput = self.branch_conv(torch.cat([x1, x4,x7,x10,x13], dim=1)) +outputs5 #,x13
        # self.featuremap1 = torch.max(outputs5, dim=1, keepdim=True)
        # self.featuremap2 = torch.mean(outputs5, dim=1, keepdim=True).detach()
        # self.featuremap1=self.featuremap1[0].detach()
        Finaloutput_last=self.last(outputs5)
        # self.featuremap5 = torch.max(Finaloutput_last, dim=1, keepdim=True)
        # self.featuremap5=self.featuremap5[0].detach()
        # self.featuremap6 = torch.mean(Finaloutput_last, dim=1, keepdim=True).detach()
        return Finaloutput_last
       # f1 = self.conv4(self.gap(inputs))
       # f2 = self.conv5(self.gap(inputs))
      #  f3 = self.conv6(self.gap(inputs))
      #  x1 = torch.mul(x1, f1)
      #  x2 = torch.mul(x2, f2)
      #  x3 = torch.mul(x3, f3) #, x3
       # outputs = self.f_conv(torch.cat([x1, x2,x3], dim=1)) + inputs

        


class HRFCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(HRFCN,self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = HRModule(num_filters)
        self.conv3 = HRModule(num_filters)
        self.conv4 = HRModule(num_filters)
        self.conv5 = HRModule(num_filters)
        
        self.last =  nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,inputs):
        x = F.relu(self.conv1(inputs))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.last(x)
        return x





class SiameseFCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(SiameseFCN,self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.last =  nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward_once(self,inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x

    def forward(self, input_1, input_2):
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)

        return output_1, output_2

class SiameseUNet(nn.Module):

    def __init__(self):
        super(SiameseUNet,self).__init__()

        self.UNet = smp.Unet(
            encoder_name='resnet18', encoder_depth=3, encoder_weights=None,
            decoder_channels=(128, 64, 64), in_channels=9, classes=len(utils.NLCD_CLASSES)
        )

    def forward_once(self,inputs):
        x = self.UNet(inputs)
        return x

    def forward(self, input_1, input_2):
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)

        return output_1, output_2

class SiameseHRNet_OCR(nn.Module):

    def __init__(self):
        super(SiameseHRNet_OCR,self).__init__()

        self.hrnet_ocr = HighResolutionNet(input_channels=9, output_channels=len(utils.NLCD_CLASSES))

    def forward_once(self,inputs):
        x = self.hrnet_ocr(inputs)
        return x

    def forward(self, input_1, input_2):
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)

        return output_1, output_2

def get_unet():
    return smp.Unet(
        encoder_name='resnet18', encoder_depth=3, encoder_weights=None,
        decoder_channels=(128, 64, 64), in_channels=4, classes=len(utils.NLCD_CLASSES)
    )

def get_deeplabv3plus():
    return smp.DeepLabV3Plus(
        encoder_name='resnet34', 
        encoder_depth=5, encoder_weights=None,
        encoder_output_stride=16, decoder_channels=256, 
        decoder_atrous_rates=(12, 24, 36), in_channels=3, 
        classes=len(utils.NLCD_CLASSES), activation=None, upsampling=4,
        aux_params=None
    )
    

def get_fcn():
    return FCN(num_input_channels=9, num_output_classes=len(utils.NLCD_CLASSES), num_filters=64)

def get_skip_fcn():
    return Skip_FCN(num_input_channels=4, num_output_classes=len(utils.NLCD_CLASSES), num_filters=64)

def get_hr_fcn():
    return HRFCN(num_input_channels=4, num_output_classes=len(utils.NLCD_CLASSES), num_filters=64)
    
def get_l2hnet():
    return L2HNet(insize=8,input_chs=128, num_output_classes=len(utils.NLCD_CLASSES))    

# def get_hrnet():
#     return HighResolutionNet(input_channels=4, output_channels=len(utils.NLCD_CLASSES))
def get_unetformer():
    return  UNetFormer(num_classes=len(utils.NLCD_CLASSES))

def get_coatnet():
    return CoAtNet_Seg(img_size=(224,224), in_channel=4, num_classes=len(utils.NLCD_CLASSES))

def get_mobilevit():
    return MobileViT_Seg(img_size=(256,256), in_channel=4, num_classes=len(utils.NLCD_CLASSES))

def get_convit():
    return Convit_seg(in_chans=4, num_classes=len(utils.NLCD_CLASSES))

def get_effientvit():
    return EfficientViT(in_chans=4, num_classes=len(utils.NLCD_CLASSES))
def get_dcswin():
    return DCSwin(num_classes=len(utils.NLCD_CLASSES))


#def get_hrnet():
#    return smp.DeepLabV3(encoder_name='resnet101', encoder_depth=5, 
#    encoder_weights=None, decoder_channels=256,
#    in_channels=3, classes=len(utils.NLCD_CLASSES), activation=None, 
#    upsampling=8, aux_params=None)
#def get_deeplabv3plus():
#    return DeepLabV3Plus(
#        input_chs=3,
#        n_classes=len(utils.NLCD_CLASSES),
#        n_blocks=[3, 4, 23, 3],
#        atrous_rates=[6, 12, 18],
#        multi_grids=[1, 2, 4],
#        output_stride=16,
#    )

# def get_refinenet():
#     return rf101(input_chs=4, num_classes=len(utils.NLCD_CLASSES))

def get_siamese_fcn():
    return SiameseFCN(num_input_channels=4, num_output_classes=len(utils.NLCD_CLASSES), num_filters=64)

def get_siamese_unet():
    return SiameseUNet()

def get_siamese_hrnet_ocr():
    return SiameseHRNet_OCR()
