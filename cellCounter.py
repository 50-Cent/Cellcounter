#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:37:24 2020

@author: Tamal Batabyal
"""
#%% _______________ DO NOT CHANGE this part____________________ %%

import sys
import os
import torch
import random
import numpy as np
import copy
import cv2
import shutil
import glob
import warnings


#%%
mssg = '\n\nThis is Kapur Lab property@University of Virginia.\nIf you need any assistance, please email John Williamson (jmw6z@virginia.edu) or Tamal Batabyal (tb2ea@virginia.edu)\n\n'
print(mssg)


#%% Path selection+ directory creation

package_path = os.getcwd() 
person_name = input('Please ENTER your name:: \n')
if person_name:
    folder_path = os.path.join(package_path, person_name)
else:
    folder_path = os.path.join(package_path, 'LocalTmp')

if not os.path.exists(folder_path):
    os.mkdir(folder_path)
    
    
data_path = os.path.join(folder_path,'Data')
if not os.path.exists(data_path):
    os.mkdir(data_path)
    
    
MIPimg_dir = os.path.join(folder_path,'MIPimages')
if not os.path.exists(MIPimg_dir):
    os.mkdir(MIPimg_dir)
    

DepthLoc_dir = os.path.join(folder_path, 'DepthData')
if not os.path.exists(DepthLoc_dir):
    os.mkdir(DepthLoc_dir)

#%%
depthStatus = 0

#%%   Load images 

mssg = '\n... There are 3 options to load the images.'
print(mssg)
mssg = '\n... Tiffstack (from .nd2 file by ImageJ) || precomputed MIP image and depthmap || single slice \n\n'
print(mssg)


ans = input('\n Are you loading MIP image (click 1)/ image (click 2)/ z-stack (ImageJ) (click 3)?\n')
if ans == '3':
    
    print('\n.. Load your Tiffstack in Data directory.\n')
    data_folders = os.listdir(data_path)
    if len(data_folders)==0:
        sys.exit("No image is found :: Load images from ImageJ")  
  
    print('\n')           
    
    if len(os.listdir(MIPimg_dir)) > 0:
       warnings.warn('Old MIP images may be present. Please check..')    
       
    print('\n')   
        
    if len(os.listdir(DepthLoc_dir)) > 0:
       warnings.warn('Old Depth data are already present. You may want to DELETE it ')  



    print('\n')

  #  MIP image formation

    for folder in data_folders:
        img_list = os.listdir(os.path.join(data_path,folder))
        img_list = [f for f in img_list if 'c002' in f]
        if not img_list:
            sys.exit("No channel 2 (tdTomato)  image is found:: Program is aborting")
        
        Ant2Pos = [f.split('_') for f in img_list]
        Ant2Pos = [f[1] for f in Ant2Pos]
        Ant2Pos = [int(f[1:]) for f in Ant2Pos]
        Ant2Pos = np.argsort(Ant2Pos)
        img_list = list( np.array(img_list)[Ant2Pos.astype(int)] )
        
        first_img = cv2.imread(os.path.join(data_path,folder,img_list[0]),0)
        img_dim = np.array(first_img.shape)
        extra_row, extra_col = 2**int(np.ceil(np.log2(img_dim[0])))-img_dim[0], 2**int(np.ceil(np.log2(img_dim[1])))-img_dim[1]
        # + : Add , - : Delete  
        
        count = 0
        print('\n___ Check whether the images are serially appeared below___\n')
        for img_name in img_list:        
            print(img_name)
            img = cv2.imread(os.path.join(data_path,folder,img_name) ,0)
            img = np.float_(img)
            if extra_row > 0:            
                img = np.concatenate( (img,np.zeros( (extra_row,img_dim[1]) )), axis=0 )
                row_dim = img_dim[0]+ extra_row
            elif extra_row < 0:            
                img = img[:img_dim[0]+extra_row,:]
                row_dim = img_dim[0]+extra_row
            
            
            if extra_col > 0:            
                img = np.concatenate( (img, np.zeros( (row_dim,extra_col) ) ), axis=1 )
                col_dim = img_dim[1]+ extra_col
            elif extra_col < 0:            
                img = img[:,:img_dim[1]+extra_col]
                col_dim = img_dim[1]+extra_col
            
            count+=1        
            if count == 1:
                img_stack = np.empty((0,row_dim,col_dim))
                
            img_stack = np.concatenate((img_stack, np.array([img])),axis=0)        
        
        #max intensity projection
        img = np.amax(img_stack,axis=0)
        depth_img = np.argmax(img_stack, axis=0)
        img_name = 'MIP_'+folder+'.tif'
        img_name_w_path = os.path.join(MIPimg_dir, img_name)
        cv2.imwrite(img_name_w_path, np.uint8(img))
        filename = open(os.path.join(DepthLoc_dir, 'Loc_'+folder+'.txt'),'w')
        np.savetxt(filename,depth_img.astype(int),fmt='%i', delimiter=" ")    #UPDATE step 
        depthStatus = 1
        
        del img_stack, img, depth_img, first_img
    
    print('_____ Maximum Intensity Images are created_____\n')
    print('_____ Depth maps are saved _____\n')    
    
    ans = input('DO you want to DELETE the files that you import from ImageJ? y/N \n')
    if ans== 'y':
        shutil.rmtree(data_path)
        os.mkdir(data_path)
        


elif ans == '1':
    print('\n Load the MIP images in the MIPimages directory..\n')
    print('\n Load the depth map in the DepthData directory..\n')
    if len(os.listdir(DepthLoc_dir)) > 0: 
        depthStatus = 1
    else:
        sys.exit("Please load the depthmap..\n ") 
    
elif ans == '2':
    print('\n Load the Slice image in the MIPimages directory..\n')
    #print('\n Load the depth map in the DepthData directory..\n')
    
else:
    sys.exit("Make proper selection (1, 2 or 3) to load the image(s)..\n ") 


print('\n\n____ Proceeding to CELL LABELLING____ \n\n')



#%% User dialogue

print('\n\n Default cellcounter:: tdTomato-default \n')
print('.... IF you want Default, Type 0 and press ENTER ... \n ')

print(' _____ Other cellcounter options _____ \n')
print('1. tdTomato-1 \n')
print('2. tdTomato-3 (Ext) \n')
print('4. dapi+tdTomato-1 \n')
print('5. dapi+tdTomato-2 \n')
print('6. NeuN-1 \n')
print('7. NeuN-2 \n')

choice = input('\n TYPE the number and then press ENTER..\n ')


#%% import modules

cellcounter_path = os.path.join(package_path, 'CellCounter')
sys.path.append(cellcounter_path)
from MyClass import buildNetwork


#%% Call object
cellObj = buildNetwork()

cellObj.root_path = os.path.join(cellcounter_path,'tdTomato_data')
cellObj.dup_root_path = os.path.join(cellObj.root_path, cellObj.dup_root_path) 
cellObj.write_path = os.path.join(folder_path,'tmp_res_path') 

if os.path.exists(cellObj.write_path): #reset 
   shutil.rmtree(cellObj.write_path)

cellObj.which_net_to_use = 'Best' 
cellObj.best_test_weight_name_status=True

if choice == '1':
    cellObj.best_test_weight_name_if_changed = os.path.join(cellcounter_path,'PGBnSQwatn_4_best_weights')
    print('\n___ Choice 1 (tdTomato) loaded____\n')
elif choice == '2':
    cellObj.best_test_weight_name_if_changed = os.path.join(cellcounter_path, 'PGBnSQwatn_7_7620_p1_best_weights')
    print('\n___ Choice 3 (tdTomato) loaded____\n')
elif choice == '3':
    cellObj.best_test_weight_name_if_changed = os.path.join(cellcounter_path, 'PGBnSQwatn_mixed_2_best_weights')
    print('\n___ Choice 4 (Dapi-tdTomato) loaded____\n')
elif choice == '4':
    cellObj.best_test_weight_name_if_changed = os.path.join(cellcounter_path, 'PGBnSQwatn_mixed_3_best_weights')
    print('\n___ Choice 5 (Dapi-tdTomato) loaded____\n')
elif choice == '5':
    cellObj.best_test_weight_name_if_changed = os.path.join(cellcounter_path, 'PGBnSQwatn_NeuN_5_best_weights')    
    print('\n___ Choice 6 (NeuN) loaded____\n')  
elif choice == '6':
    cellObj.best_test_weight_name_if_changed = os.path.join(cellcounter_path, 'PGBnSQwatn_NeuN_6_best_weights')    
    print('\n___ Choice 7 (NeuN) loaded____\n')     
else:
    cellObj.best_test_weight_name_if_changed = os.path.join(cellcounter_path, 'PGBnSQwatn_5_best_weights')
    print('\n___ Default cellcounter (tdTomato) loaded____\n')

cellObj.manual_test_path_status = True
cellObj.manual_test_path = cellObj.write_path # results on 64*64 patches 


tmp_test_path = os.path.join(cellObj.root_path,cellObj.test_dataDir) # split of an image into 64*64
if os.path.exists(tmp_test_path): #reset 
   shutil.rmtree(tmp_test_path)

    
write_final_path = os.path.join(folder_path,'Final_result')

    
list_img = os.listdir(MIPimg_dir)
    
for k in np.arange(len(list_img)):
    if not os.path.exists(tmp_test_path):
        os.mkdir(tmp_test_path)
    
    if not os.path.exists(write_final_path):
        os.mkdir(write_final_path)
        
    if not os.path.exists(cellObj.manual_test_path):
        os.mkdir(cellObj.manual_test_path)
    
    
    img = list_img[k]
    img_name_w_path = os.path.join(MIPimg_dir, img)  #read images
    ip_img = cv2.imread(img_name_w_path,0)
    M, N = ip_img.shape[0], ip_img.shape[1]
    ip_img = np.float_(ip_img)
    if M % 64 != 0:
        to_del = np.arange((M % 64))
        ip_img = np.delete(ip_img,to_del,0)
        M -= M % 64
    
    if N % 64 != 0:
        to_del = np.arange((N % 64))
        ip_img = np.delete(ip_img,to_del,1)
        N -= N % 64 
    
       
    xIdx = np.arange(0, M+1, 64)
    yIdx = np.arange(0, N+1, 64)
    
    for ii in np.arange(len(xIdx)-1):               
        for jj in np.arange(len(yIdx)-1):
            blck = ip_img[xIdx[ii]:xIdx[ii+1],yIdx[jj]:yIdx[jj+1]]
            img_name = tmp_test_path + '/' + str(ii)+'_'+str(jj) + '_'+img          
            cv2.imwrite(img_name, np.uint8(blck))
        
        cellObj.detection_thresh = 0.10    
        cellObj.testNetwork()  #results of counting on patches saved in cellObj.manual_test_path
        flss = glob.glob(tmp_test_path+'/*')
        [os.remove(f) for f in flss]  
    
    # Now tmp_test_path is empty
    #shutil.rmtree(tmp_path)   #DELETE where 64*64 patches are kept
    
    out_img = np.zeros((M,N,3))
    img = img.split('.')[0]
    for ii in np.arange(len(xIdx)-1):
        for jj in np.arange(len(yIdx)-1):
            imgg = cellObj.manual_test_path + '/' + str(ii)+'_'+str(jj) + '_'+img+'_color.tif'
            img_patch = cv2.imread(imgg,1)
            out_img[ xIdx[ii]:xIdx[ii+1],yIdx[jj]:yIdx[jj+1],:] = np.float_(img_patch)            
    
    cv2.imwrite(os.path.join(write_final_path,'count_'+img+'.tif'), np.uint8(out_img))  
    shutil.rmtree(cellObj.manual_test_path)
    
    print('{} labeling completed....\n'.format(img))
    
print('\n\n All the images are done. Congratulations.!!\n\n')
print('___ COVID19: WEAR face mask :: WASH your face and hands :: KEEP 6 ft distance :: timely REPORT__\n\n')

 


   
        
        
            
            
        
    
    
     

