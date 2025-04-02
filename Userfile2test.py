# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:54:30 2025

@author: Tamal
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from MyClass import buildNetwork
from ConfigureDataset import NeuronDataSet
from torch.utils.data import DataLoader
import os
# =============================================================================
from myNetwork import CellCounter
from torchinfo import summary
# =============================================================================
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label
import cv2
from matplotlib import pyplot as plt

#%%
def getCenterList(img):      
    is_peak = peak_local_max(img,min_distance=2,threshold_abs=0.24,exclude_border=False)  
    labels = label(is_peak)[0]
    center_list = np.ceil(center_of_mass(is_peak, labels, range(1, np.max(labels)+1)))
    del is_peak, labels 
    
    return center_list

def my_gray2color(my_img):
    ip_dim= my_img.shape
    op_img = np.zeros((ip_dim[0],ip_dim[1],3))
    op_img[:,:,0] = np.float_(my_img)
    op_img[:,:,1] = np.float_(my_img)
    op_img[:,:,2] = np.float_(my_img)
    op_img = np.uint8(op_img)
    
    del ip_dim
    return op_img
#%% Train a network if needed
#%% Pre-trained configurations:
#%%      PGBnSQwatn_4_best_weights.pt, PGBnSQwatn_5_best_weights (for tdTomato images)
#%%      PGBnSQwatn_NeuN_5_best_weights.pt, PGBnSQwatn_NeuN_6_best_weights.pt (for NeuN images) 


mynet = buildNetwork() 
net   = mynet.network
name_best_training_wgt = "PGBnSQwatn_4_best_weights" + ".pt" 
wgt = os.path.join('CellCounter',name_best_training_wgt)
net.load_state_dict(torch.load(wgt))
dvce = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(dvce)





#%%
testdataobj = NeuronDataSet(root_path='Dataroot', dataDir='testData',\
                            regressLabel='test_regressLabel', dottedLabel='test_dottedLabel', \
                                regStatus=False, dotStatus=False, augStatus=False,\
                                    num_batch=1, rotationAng=0)
    

dataloader = DataLoader(testdataobj, batch_size=1, shuffle=False)

#%%
for data in dataloader:
    inputs, reglabels, dotlabels, id_list = data['image'], data['regress'], data['dotted'], data['id']
    org_idx = id_list[0].split('.')[0] 
    print(org_idx)
    original_img = cv2.imread(os.path.join('Dataroot','testData', org_idx+'.tif'),0)  #check
    
    inputs = inputs.to(dvce,dtype=torch.float)            
    outputs = net(inputs)        
    outputs = outputs.to('cpu')
    outputs = np.squeeze(outputs.detach().numpy(), axis=1) 
    outputs = np.squeeze(outputs,axis=0)
    
    #plt.rcParams["figure.figsize"] = [7.00, 3.50]
    #plt.rcParams["figure.autolayout"] = True
    data = np.float_(outputs)
    plt.imshow(data)
    plt.show()
# =============================================================================
#     center_list = getCenterList(outputs)
#     center_list = np.array(center_list)
#     color_input = my_gray2color(original_img)
#     xx = np.array([])
#     yy = np.array([])
#     if center_list.size != 0:
#         xx,yy = center_list[:,0], center_list[:,1]           
#         center_mask = np.zeros(original_img.shape)
#         center_mask[xx.astype(np.int_), yy.astype(np.int_)] = [1]
#         color_input[center_mask.astype(bool)] = [0, 0, 255]
#     
#     cv2.imwrite('testresult'+'/'+org_idx+'_color.tif',color_input)
# =============================================================================
