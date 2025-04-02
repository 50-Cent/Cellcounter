# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:43:41 2020

@author: TamalVIVA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


    
        

class practiceNet(nn.Module):
    def __init__(self):
        super(practiceNet,self).__init__()
        self.name = 'practiceNet'
        self.conv1 = nn.Conv2d(1,8,3, padding=(1,1))
        self.conv2 = nn.Conv2d(8,16,3,padding=(1,1))
        self.conv3 = nn.Conv2d(16,64,5,padding=(2,2))
        self.conv4 = nn.Conv2d(64,8,3,padding=(1,1))
        self.conv5 = nn.Conv2d(8,1,3,padding=(1,1))
        
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        
        
        
    def forward(self, x):
        #print(x)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        b = x.view(x.shape[0],x.shape[1], x.shape[2]*x.shape[3])
        c = torch.transpose(b,1,2)
        b = F.softmax(c@b, dim=2)
        b = torch.transpose(b@c,1,2)
        x = b.view(x.shape)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        return x


#%% Final architecture (Improved)
        
class piggyBackNeXtCSLD(nn.Module):    
    def __init__(self):
        super(piggyBackNeXtCSLD,self).__init__()
        self.name ='pggBNCA'
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)        
        #initial set up (64*64)
        self.conv_1_33 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_33_bn = nn.BatchNorm2d(16)
        self.conv_1_55 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv1_55_bn = nn.BatchNorm2d(16)
        self.conv_1_77 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv1_77_bn = nn.BatchNorm2d(16)
        self.conv_1_99 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.conv1_99_bn = nn.BatchNorm2d(16)

        # Level-1 "piggyback and squeeze" layers (downsample) (32*32 op)
        self.conv_2_33 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv2_33_bn = nn.BatchNorm2d(128)
        
        # consolidation
        self.conv_3_33 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_33_bn = nn.BatchNorm2d(128)
        
        # Level-2 "piggyback and squeeze" layers (downsample) (16*16 op)
        self.conv_4_33 = nn.Conv2d(in_channels=64+128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4_33_bn = nn.BatchNorm2d(256) 
        
        # consolidation
        self.conv_5_33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_33_bn = nn.BatchNorm2d(256)
        
        # Level-3 "piggyback and squeeze" layers (downsample) (8*8 op)      
        self.conv_6_33 = nn.Conv2d(in_channels=256+128, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv6_33_bn = nn.BatchNorm2d(512)
        
        # consolidation
        self.conv_7_33 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7_33_bn = nn.BatchNorm2d(512)
        
        # Level-4 "piggyback and squeeze" layers (8*8 op)
        self.conv_8_33 = nn.Conv2d(in_channels=512+256, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv8_33_bn = nn.BatchNorm2d(1024)
        
       
#         Last convolution block w upsample
        self.conv_1024_128 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv1024_128_bn = nn.BatchNorm2d(128)
        self.conv_128_1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv128_1_bn = nn.BatchNorm2d(1)
        
        
        
        torch.nn.init.kaiming_normal_(self.conv_1_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_1_55.weight)
        torch.nn.init.kaiming_normal_(self.conv_1_77.weight)
        torch.nn.init.kaiming_normal_(self.conv_1_99.weight)
        torch.nn.init.kaiming_normal_(self.conv_2_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_3_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_4_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_5_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_6_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_7_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_8_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_1024_128.weight)
        torch.nn.init.kaiming_normal_(self.conv_128_1.weight)        

        
    def forward(self,x):
        x1 = self.conv1_33_bn(F.relu(self.conv_1_33(x)))
        x2 = self.conv1_55_bn(F.relu(self.conv_1_55(x)))
        x3 = self.conv1_77_bn(F.relu(self.conv_1_77(x)))
        x4 = self.conv1_99_bn(F.relu(self.conv_1_99(x)))
        x = torch.cat((x1,x2,x3,x4),dim=1)
        
        x1 = self.conv2_33_bn(F.relu(self.conv_2_33(x)))      # 32*32
        x1 = self.conv3_33_bn(F.relu(self.conv_3_33(x1)))     # consolidation
        x  = self.max_pool(x)                                 # 64*64 --> 32*32        
        x  = torch.cat((x,x1),dim=1)                          #piggyback
        
        x  = self.conv4_33_bn(F.relu(self.conv_4_33(x)))     # 16*16
        x  = self.conv5_33_bn(F.relu(self.conv_5_33(x)))     # consolidation
        x1 = self.max_pool(x1)                               # 16*16
        # b = x1.view(x1.shape[0],x1.shape[1], x1.shape[2]*x1.shape[3]) # Attention
        # c = torch.transpose(b,1,2)
        # b = F.softmax(c@b, dim=2)
        # b = torch.transpose(b@c,1,2)
        # x1 = b.view(x1.shape)                                         # Attention
        x1 = torch.cat((x,x1),dim=1)                         # piggyback
        
        x1 = self.conv6_33_bn(F.relu(self.conv_6_33(x1)))    # 8*8
        x1 = self.conv7_33_bn(F.relu(self.conv_7_33(x1)))    # consolidation
        x  = self.max_pool(x)                                 # 16*16 --> 8*8 
        b = x.view(x.shape[0],x.shape[1], x.shape[2]*x.shape[3]) #Attention
        c = torch.transpose(b,1,2)
        b = F.softmax(c@b, dim=2)
        b = torch.transpose(b@c,1,2)
        x = b.view(x.shape)                                     #Attention     
        x  = torch.cat((x,x1),dim=1)                          #piggyback
        
        x  = self.conv8_33_bn(F.relu(self.conv_8_33(x)))      # 1024*8*8
        
        x = F.interpolate(x, scale_factor=4, mode='nearest')  #16*16                        
        x = self.conv1024_128_bn(F.relu(self.conv_1024_128(x)))  
        x = F.interpolate(x, scale_factor=2, mode='nearest')  #64*64
        x = self.conv128_1_bn(F.relu(self.conv_128_1(x)))      
        out = F.relu(x)
        
        del x1,x2,x3,x4
        # del b,c
        
        return out

#%% 

class CellCounter(nn.Module):
    def __init__(self):
        super(CellCounter,self).__init__()
        self.name = 'PGBnSQwatn'
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        
        #UP
        self.conv_1_33 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_33_bn = nn.BatchNorm2d(32)
        
        self.conv_2_33 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_33_bn = nn.BatchNorm2d(128)
        
        self.conv_3_33 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1)#downsample
        self.conv3_33_bn = nn.BatchNorm2d(32)
        
        self.conv_4_33 = nn.Conv2d(in_channels=32+32, out_channels=512, kernel_size=3, stride=1, padding=1) 
        self.conv4_33_bn = nn.BatchNorm2d(512)
        
        self.conv_5_33 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, stride=2, padding=1) #downsample
        self.conv5_33_bn = nn.BatchNorm2d(32)
        
        self.conv_6_33 = nn.Conv2d(in_channels=32+32, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv6_33_bn = nn.BatchNorm2d(1024)
        
        self.conv_7_33 = nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=3, stride=2, padding=1) #downsample
        self.conv7_33_bn = nn.BatchNorm2d(32)
        
        
        #DOWN
        self.conv_8_33 = nn.Conv2d(in_channels=32+32, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv8_33_bn = nn.BatchNorm2d(32)       
        
        self.conv_9_33 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1) 
        self.conv9_33_bn = nn.BatchNorm2d(16)
        
        self.conv_10_33 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1) 
        self.conv10_33_bn = nn.BatchNorm2d(8)
        
        self.conv_11_33 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1) 
        self.conv11_33_bn = nn.BatchNorm2d(4)
        
        self.conv_12_33 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1) 
        self.conv12_33_bn = nn.BatchNorm2d(2)
        
        self.conv_13_33 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1) 
        self.conv13_33_bn = nn.BatchNorm2d(1)
        
        
        torch.nn.init.kaiming_normal_(self.conv_1_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_2_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_3_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_4_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_5_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_6_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_7_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_8_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_9_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_10_33.weight) 
        torch.nn.init.kaiming_normal_(self.conv_11_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_12_33.weight)
        torch.nn.init.kaiming_normal_(self.conv_13_33.weight) 
    
    
    def forward(self,x):
        x = self.conv1_33_bn(F.relu(self.conv_1_33(x)))
        x1 = self.conv2_33_bn(F.relu(self.conv_2_33(x)))
        
        x1 = self.conv3_33_bn(F.relu(self.conv_3_33(x1))) #downsample
        x = self.max_pool(x) 
        x = torch.cat((x,x1),dim=1)                       #piggyback
        
        x = self.conv4_33_bn(F.relu(self.conv_4_33(x))) 
        x = self.conv5_33_bn(F.relu(self.conv_5_33(x)))   #downsample
        x1 = self.max_pool(x1)                           #insert ATN here
        x1 = torch.cat((x1,x),dim=1)
        
        x1 = self.conv6_33_bn(F.relu(self.conv_6_33(x1)))
        b = x1.view(x1.shape[0],x1.shape[1], x1.shape[2]*x1.shape[3]) #Attention
        c = torch.transpose(b,1,2)
        b = F.softmax(c@b, dim=2)
        b = torch.transpose(b@c,1,2)                                  #Attention
        
        
        x1 = b.view(x1.shape)
        x1 = self.conv7_33_bn(F.relu(self.conv_7_33(x1))) #downsample
        x = self.max_pool(x)                           
        x = torch.cat((x,x1),dim=1)
        
        x = self.conv8_33_bn(F.relu(self.conv_8_33(x))) 
        x = F.interpolate(x, scale_factor=2, mode='nearest') #upsample
        x = self.conv9_33_bn(F.relu(self.conv_9_33(x)))
        x = F.interpolate(x, scale_factor=2, mode='nearest') #upsample
        x = self.conv10_33_bn(F.relu(self.conv_10_33(x)))
        x = F.interpolate(x, scale_factor=2, mode='nearest') #upsample
        x = self.conv11_33_bn(F.relu(self.conv_11_33(x)))
        x = self.conv12_33_bn(F.relu(self.conv_12_33(x)))
        x = self.conv13_33_bn(F.relu(self.conv_13_33(x)))
        
        
        out = F.relu(x)
        
        del x1
        del b,c
        
        return out

#%% 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tmp = CellCounter().to(device)
#summary(tmp,(1,64,64))


        
        