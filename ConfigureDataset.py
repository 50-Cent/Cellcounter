# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:30:56 2020

@author: TamalVIVA
"""

import torch
import numpy as np
# import copy
import os
# import pandas as pd
from torch.utils.data import Dataset
# from torchvision import transforms, utils
import cv2
import random
# import scipy as scp
from scipy.ndimage.measurements import center_of_mass, label
from skimage.feature import peak_local_max
from scipy import ndimage

#%%
class NeuronDataSet(Dataset):
    def __init__(self, root_path,dataDir,regressLabel,dottedLabel,regStatus,dotStatus,augStatus,num_batch,rotationAng):       
        self.dataRootPath = root_path
        self.dataDir = dataDir
        #self.testDataDir = 'testDir'
        #self.validationDataDir = 'valdDir'
        self.regressLabel = regressLabel
        #self.validationRegressLabel = 'trainValdLabel'
        self.dottedLabel = dottedLabel
        #self.valdLabel = 'valdLabel'
        self.isRegress = regStatus
        self.isDot = dotStatus        
        self.dataAugment = augStatus
        #self.dataScale = False
        #self.dataRotate = True
        #self.dataTranslate = False
        #self.train2vald_ratio = 0.9
        self.imgnames = os.listdir(os.path.join(self.dataRootPath,self.dataDir))
        self.detection_thresh = 0.6
        self.reg_threshold = 0.1
        self.img_threshold = 2
        self.no_batch = num_batch
        self.rotationAngle = rotationAng
        
        
    
    def getCenterList(self,img):      
        is_peak = peak_local_max(img,min_distance=1,threshold_abs=self.detection_thresh,exclude_border=False,indices=False)  
        labels = label(is_peak)[0]
        center_list = np.ceil(center_of_mass(is_peak, labels, range(1, np.max(labels)+1)))
        del is_peak, labels 
        
        return center_list
        
    
    def __len__(self):
        return len(self.imgnames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name_w_path = os.path.join(self.dataRootPath,self.dataDir,self.imgnames[idx])
       # print(img_name_w_path)
        ip_img = cv2.imread(img_name_w_path,0)
        ip_img = np.float_(ip_img)
        
        if self.isRegress:
            reg_name_w_path = os.path.join(self.dataRootPath,self.regressLabel,self.imgnames[idx])
            reg_img = cv2.imread(reg_name_w_path,0)
            reg_img = np.float_(reg_img)
            reg_img/=255
        else:
            reg_img = []
            
        
        dot_list = []
        if self.isDot:
            if self.no_batch==1:   # dot_list NOT iincluded if batch size > 1
                org_idx = self.imgnames[idx]
                org_idx = org_idx.split('.')[0]
                dot_name_w_path = os.path.join(self.dataRootPath,self.dottedLabel, org_idx+'.txt')
                with open(dot_name_w_path) as fID:
                    dot_list = fID.read().replace('\n', ' ')
                dot_list = dot_list.split(' ')
                dot_list = [x for x in dot_list if x!='']
                dot_list = [int(x) for x in dot_list]
                dot_list = np.array(dot_list)
                dot_list = dot_list.reshape((-1,2)) 
                dot_list -= 1   #locations are from MATLAB
        
        
        
        sample = {'image': ip_img, 'regress': reg_img, 'dotted': dot_list, 'id':self.imgnames[idx]}
        
        if self.dataAugment:
            # print(sample['dotted'])
            #sample = self.augmentationRoutine(sample)
            # ri = random.randint(0, 360)
            #print(self.rotationAngle)
            ip_img = ndimage.rotate(ip_img,self.rotationAngle, reshape=False)
            ip_img[np.where(ip_img<self.img_threshold)] = 0 
            reg_img = ndimage.rotate(reg_img,self.rotationAngle,reshape=False)
            reg_img[np.where(reg_img<self.reg_threshold)]=0  
            ll = np.array(sample['dotted'])
            if ll.shape[0]>0:
                center_listt = self.getCenterList(reg_img)
            else:
                center_listt = [] 
                
            if np.max(ip_img) > np.min(ip_img):            
                enhanced_image = (ip_img - np.min(ip_img)) / (np.max(ip_img) - np.min(ip_img))
            else:
                enhanced_image = 0*ip_img
            
            # print(center_listt)
            enhanced_image = np.expand_dims(enhanced_image, axis=0).astype(np.float64)
            reg_img = np.expand_dims(reg_img, axis=0).astype(np.float64) 
            
            sample = {'image': enhanced_image, 'regress': reg_img, 'dotted': center_listt, 'id':sample['id']}
            
        else:
            if np.max(ip_img) > np.min(ip_img):            
                enhanced_image = (ip_img - np.min(ip_img)) / (np.max(ip_img) - np.min(ip_img))
            else:
                enhanced_image = 0*ip_img
            
            enhanced_image = np.expand_dims(enhanced_image, axis=0).astype(np.float64)
            reg_img = np.expand_dims(reg_img, axis=0).astype(np.float64) 
            
            sample = {'image': enhanced_image, 'regress': reg_img, 'dotted': dot_list, 'id':self.imgnames[idx]}
        
        return sample
#%%  
   
    # def augmentationRoutine(self,sample):
    #     img = sample['image']
    #     reg_img = sample['regress']        
    #     # select only random integer as a rotation
    #     ri = random.randint(0, 360)
    #     img = ndimage.rotate(img,ri, reshape=False)
    #     img[np.where(img<self.img_threshold)] = 0                   # get rid of unwanted exponent values
    #     reg_img = ndimage.rotate(reg_img,ri,reshape=False)
    #     reg_img[np.where(reg_img<self.reg_threshold)]=0             # get rid of unwanted exponent values
        
    #     ll = np.array(sample['dotted'])
    #     if ll.shape[0]>0:
    #         center_listt = self.getCenterList(reg_img)
    #     else:
    #         center_listt = []            
        
    #     #print(center_list)
    #     if np.max(img) > np.min(img):            
    #         enhanced_image = (img - np.min(img)) / (np.max(img) - np.min(img))
    #     else:
    #         enhanced_image = 0*img
        
    #     enhanced_image = np.expand_dims(enhanced_image, axis=0).astype(np.float64)
    #     reg_img = np.expand_dims(reg_img, axis=0).astype(np.float64) 
            
    #     sample = {'image': enhanced_image, 'regress': reg_img, 'dotted': center_listt, 'id':sample['id']}
        
        
    #     return sample
        

#%%

# for i_b, data_b in enumerate(sampledataloader):
#     print(i_b, data_b['image'].size())
        
# for data in sampledataloader:
#     print(data['regress'].shape)       
        
        
        
        
        