# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:31:00 2020

@author: TamalVIVA
"""

#%% Modules
import torch
import numpy as np
# import copy
import os
# import pandas as pd
# from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
# from torchvision import transforms, utils
# import random
import torch.optim as optim
import torch.nn as nn
import cv2, time
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label

#%% Custom modules
# from myNetwork import practiceNet

from myNetwork import CellCounter
from ConfigureDataset import NeuronDataSet
#from FCNN import FCNNA

 

#%% 

class buildNetwork():
    def __init__(self):
        self.network = CellCounter()  
        self.which_net_to_use = 'Best' #options:: "Best" (for testing), "Current" (during training only)
        self.save_best_config_net = True
        self.write_path = 'write_path'
        self.dataTrainAugment = True  
        self.numRotation = 16
        self.rotationSet = np.array([0,45,90,135,18,225,270,315])
        self.trainIdxStep = 2
        self.defaultBatchSize = 2           #change gradually to 4, 8 and 16 and later for larger batches 32 and 64
        
        self.learnrate = 0.001
        self.momentum_coeff = 0.9
        self.weight_decay = 0.05
        
        self.data_loading_workers_num = 4   #change
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_thresh = 0.29  #important param
        self.num_epochs = 2
        
        self.update_status = False
        self.epoch_update_step = 3
        self.epoch_update_start = 10 
        
        self.img_coding_scheme = 'repel'
        self.best_test_weight_name_status = True #Default
        self.best_test_weight_name_if_changed = 'PGBnSQwatn_5_best_weights' #user 
        self.repel_decay = 0.8
        self.repel_thresh = 9
        self.intensity_false_pos_rejection = 0.7
        
        self.root_path = 'Dataroot'
        self.train_dataDir = 'trainData'
        self.train_regressLabel = 'train_regressLabel'
        self.train_dottedLabel = 'train_dottedLabel'
        self.val_dataDir = 'valData'
        self.val_regressLabel = 'val_regressLabel'
        self.val_dottedLabel = 'val_dottedLabel'        
        self.test_dataDir = 'testData'
        self.test_regressLabel = 'test_regressLabel'
        self.test_dottedLabel = 'test_dottedLabel'         
        self.dup_root_path = 'DupDataroot'
        
        self.manual_count_available = False
        self.true_labeled_cell = 1
        self.true_unlabeled_cell = 1
        
        self.training_phase_indx = '0'
        self.load_best_training_weights = False
        self.name_best_training_wgt_available = False
        self.name_best_training_wgt = 'Sample_weight'
        self.result_dir_name = 'Result_1'
        self.manual_test_path_status = True #user
        self.manual_test_path = 'testresult'  # user
        
        
        
        
## auxiliary functions       
        
    def getCenterList(self,img):      
        is_peak = peak_local_max(img,min_distance=2,threshold_abs=self.detection_thresh,exclude_border=False,indices=False)  
        labels = label(is_peak)[0]
        center_list = np.ceil(center_of_mass(is_peak, labels, range(1, np.max(labels)+1)))
        del is_peak, labels 
        
        return center_list
    
    @staticmethod
    def getClosestneighbor(elem, detectedLoc):
        max_scale = 3       
        elem = list(elem)
        detect_elem = np.array([])
        detectedLoc = list(detectedLoc)
        detectedLoc = [list(i) for i in detectedLoc]      
        scl = 0        
        while scl<=max_scale:
            if scl==0:
                ngh = [elem]
                scl+=1
            else:
                m = [[x,y] for x in np.arange(-scl+1,scl) for y in np.arange(-scl+1,scl)]
                scl+=1
                s = [[x,y] for x in np.arange(-scl+1,scl) for y in np.arange(-scl+1,scl)]
                ms = [loc for loc in s if loc not in m]
                ngh = list(np.array(ms)+np.array(elem))   #np.array
                ngh = [list(i) for i in ngh] 
                
            common_elem = [entry for entry in ngh if entry in detectedLoc]
            if len(common_elem)>0:
                detect_elem = common_elem[0]
                break
        return np.array(detect_elem)
        
    
    def getStatistics(self,actualLoc,detectedLoc):        
        actualStatus = np.zeros(len(actualLoc))
        detectedStatus = np.zeros(len(detectedLoc))           
        for elem in np.arange(len(actualLoc)):
            common_elem = self.getClosestneighbor(actualLoc[elem,:],detectedLoc)
            if len(common_elem) > 0: # discard multiple count
                if actualStatus[elem]==0:
                    actualStatus[elem]=1
                    indxx = [loc for loc in np.arange(len(detectedLoc)) \
                             if np.array_equal(detectedLoc[loc],common_elem)]
                    detectedStatus[indxx] = 1
        
        tp = np.sum(actualStatus)        
        fn = len(actualLoc)-tp
        fp = len(detectedStatus)-np.sum(detectedStatus) 
        
        
#        print(actualLoc)
#        print('\n')
#        print(detectedLoc)
        #print((tp,fn,fp, len(detectedStatus)))
#        precision_val = tp/(tp+fp+0.0001)
#        recall_val    = tp/(tp+fn+0.0001)
#        f1_val = 2*tp/(2*tp+fp+fn+0.0001)
        
        # del common_elem,actualStatus,detectedStatus
        
        return tp, fn, fp
    
    @staticmethod
    def my_gray2color(my_img):
        ip_dim= my_img.shape
        op_img = np.zeros((ip_dim[0],ip_dim[1],3))
        op_img[:,:,0] = np.float_(my_img)
        op_img[:,:,1] = np.float_(my_img)
        op_img[:,:,2] = np.float_(my_img)
        op_img = np.uint8(op_img)
        
        del ip_dim
        return op_img
    
    
     
    def get_capacity(self,dataobj):
        dataloader = DataLoader(dataobj, batch_size=1, shuffle=False, \
                                num_workers=self.data_loading_workers_num)
        
        tc_all = 0.0 #estimated capacity
        rc_all = 0.0 #retention capacity
        ec_all = 0.0 #error_capacity
        self.network.to(self.device) 
        for data in dataloader:
            inputs, reglabels, dotlabels, id_list = data['image'], data['regress'], data['dotted'], data['id']
            inputs = inputs.to(self.device, dtype=torch.float)
            outputs = self.network(inputs)
            
            outputs = outputs.to('cpu')
            outputs = np.squeeze(outputs.detach().numpy(), axis=1)  
            outputs = np.squeeze(outputs,axis=0)
            inputs = inputs.to('cpu').numpy()
            inputs = np.squeeze(inputs, axis=1) 
            center_list = self.getCenterList(outputs)      #function         
            center_list = np.array(center_list)
            # dotlabels = dotlabels.to('cpu')
            # dotlabels = np.squeeze(dotlabels.detach().numpy(),axis=0)
            
            # Fetching original dot labels   #CHECK: id_list
            org_idx = id_list[0].split('.')[0]
            with open(os.path.join(self.dup_root_path, self.train_dottedLabel, org_idx+'.txt'),'r') as fID:
                img_loc = fID.read().replace('\n', ' ')
            img_loc = img_loc.split(' ')
            img_loc = [x for x in img_loc if x!='']
            img_loc = [int(x) for x in img_loc]
            img_loc = np.array(img_loc)
            img_loc = img_loc.reshape((-1,2))
            img_loc -= 1
                           
            # so we have previous and current center lists
            
            tc_manual = len(img_loc)
            tp, fn, fp = self.getStatistics(center_list,img_loc)
            tc_curr, rc_curr, ec_curr = len(center_list), tp, fp
            if tc_manual>0:
                tc_curr/=tc_manual
                rc_curr/=tc_manual
            else:
                tc_curr=0
                rc_curr=0
            if len(center_list)>0:
                ec_curr/=len(center_list)
            else:
                ec_curr = 0
            
            tc_all+=tc_curr
            rc_all+=rc_curr
            ec_all+=ec_curr
            
        return tc_all, rc_all, ec_all 
    
    
    
    def img_coding_function(self,reglabels,center_list,coding_scheme):
        label_img = np.zeros(reglabels.shape)
        
        # print(center_list)
        for i in np.arange(reglabels.shape[0]): #check
            for j in np.arange(reglabels.shape[1]):
                dstList = [np.linalg.norm(m-np.array([i, j])) for m in center_list]                
                if len(dstList) > 1:
                    dstList = np.sort(dstList)
                    D_ij = dstList[0]*((1+ dstList[0]/(dstList[1]+0.01))**2)
                else:
                    D_ij = dstList[0]
                    
                if D_ij < self.repel_thresh:
                    label_img[i,j] = 255//(1+ self.repel_decay*D_ij)
                else:
                    label_img[i,j] = 0
                    
        
        return label_img
    
    
    @staticmethod
    def searchCellLoc(X, Y, img):
        M, N = img.shape[0], img.shape[1]
       
        if (X-2>=0) and (X+3<M) and (Y-2>=0) and (Y+3<N):
            wndw = img[X-2:X+3,Y-2:Y+3]
            x,y = np.where(wndw==np.max(np.ravel(wndw)))
            #in case of tie take the first
            X, Y = X+x[0]-1, Y+y[0]-1
        
        return X, Y
    
    
    def cellLocationRefinement(self, center_list, id_list):
        center_list = center_list.astype(np.int)
        # print(center_list)
        # print('\n')
        org_idx = id_list[0].split('.')[0]
        # print(org_idx)
        # print('\n')
        with open(os.path.join(self.dup_root_path, self.train_dottedLabel, org_idx+'.txt'),'r') as fID:
            img_loc = fID.read().replace('\n', ' ')
            img_loc = img_loc.split(' ')
            img_loc = [x for x in img_loc if x!='']
            img_loc = [int(x) for x in img_loc]
            img_loc = np.array(img_loc)
            img_loc = img_loc.reshape((-1,2))
            img_loc -= 1
        fID.close()
        # print(img_loc)
        # print('\n')
        # how many of center_list match with img_loc
        actualStatus = np.zeros(len(img_loc))
        detectedStatus = np.zeros(len(center_list))
        new_center_list = np.empty((0,2))
        if len(actualStatus)>0:
            if len(center_list)>0:
                for elem in np.arange(len(img_loc)):
                    common_elem = self.getClosestneighbor(img_loc[elem,:],center_list)
                    if len(common_elem) > 0: # discard multiple count
                        if actualStatus[elem]==0:
                            actualStatus[elem]=1
                            indxx = [loc for loc in np.arange(len(center_list)) \
                                     if np.array_equal(center_list[loc],common_elem)]
                            detectedStatus[indxx] = 1
                        
            #rectification of original data location
            img = cv2.imread(os.path.join(self.root_path, self.train_dataDir,org_idx+'.tif'), 0)
            img = np.float_(img)            
            adaptive_thresh = []
            M,N = img.shape[0], img.shape[1]        
            
            #Modify locs of undetected labeled cells to be detected in the next epoch
            # print(org_idx)
            # print('\n')
            # if len(actualStatus)>0:
            for kk in np.arange(len(actualStatus)):
                if actualStatus[kk]==0:
                    locx, locy = self.searchCellLoc(img_loc[kk,0], img_loc[kk,1], img)                     
                else:
                    locx, locy = img_loc[kk,0], img_loc[kk,1]
                    
                new_center_list = np.concatenate( (new_center_list, np.array([[locx, locy]])  ) ,axis=0)
                if (locx-1 >= 0) and (locx+2 < M) and (locy-1 >= 0) and (locy+2 < N):
                    wndw = img[locx-1:locx+2,locy-1:locy+2]
                    adaptive_thresh+= list(np.ravel(wndw))
                else:
                    adaptive_thresh += [img[locx,locy]]
                        
            
            adaptive_thresh = self.intensity_false_pos_rejection*np.mean(adaptive_thresh)
                    
            #deletion of locations falling below the adaptive threshold
            
            if len(center_list)>0:
                additional_loc = np.where(detectedStatus==0)[0]      
                           
                if len(additional_loc)>0:
                    # print(additional_loc) 
                    for kk in additional_loc:
                        common_elem = self.getClosestneighbor(center_list[kk,:], img_loc)
                        if len(common_elem)==0: #not in the neighborhood of img_loc
                            wndw = img[center_list[kk,0]-1 : center_list[kk,0]+2, center_list[kk,1]-1 : center_list[kk,1]+2]
                            mean_intensity = np.mean(np.ravel(wndw))
                            if mean_intensity >= adaptive_thresh:
                                new_center_list = np.concatenate( (new_center_list, np.array([[center_list[kk,0], \
                                                                                               center_list[kk,1]]]) ) ,axis=0)
            
        
        # print(new_center_list)
        return new_center_list
                 
        
     
            
#%% main functions
        
    
    def trainNetwork(self): 
        print('... Training begins...\n')
        torch.cuda.empty_cache()
        train_loss_hist = []
        eval_loss_hist = []
        best_model = {'F1': -0.1, 'recall': -0.1, 'precision': -0.1}        
        
        #optimizer
        criterion = nn.MSELoss()        
        optimizer = optim.Adam(self.network.parameters(), lr=self.learnrate)
        
        #load previous best weights
        if self.load_best_training_weights:
            if self.name_best_training_wgt_available:
                weight_name = self.name_best_training_wgt+".pt"               
            else:                    
                vers_to_load = str(int(self.training_phase_indx)-1)           
                weight_name = self.network.name +"_" + vers_to_load+"_best_weights.pt"
                weight_name = os.path.join(self.write_path +'/' +self.network.name+'_'+self.result_dir_name, weight_name) 
            
            self.network.load_state_dict(torch.load(weight_name, weights_only=True))
            self.network.eval()
            print('Weights_loaded..\n')
        
        
        self.network.to(self.device)
              
        epoch_update_array = np.arange(self.epoch_update_start,self.num_epochs,self.epoch_update_step)
        
       
        for epoch in np.arange(self.num_epochs):
            train_loss = 0.0
            tic = time.time() 
            print('epoch {}\n'.format(epoch))
            
            print('...train on original data...\n')
            
            #Original data (no rotation :: augStatus=False)
            traindataobj = NeuronDataSet(root_path=self.root_path, dataDir=self.train_dataDir, \
                                          regressLabel=self.train_regressLabel, dottedLabel=self.train_dottedLabel,\
                                              regStatus=True, dotStatus=True, augStatus=False,\
                                                  num_batch=self.defaultBatchSize, \
                                                      rotationAng=0) # dot_list NOT included if batch > 1
            
            lenData = len(traindataobj)
            dataloader = DataLoader(traindataobj, batch_size=self.defaultBatchSize, \
                                    shuffle=True)
            print('accessing GPU \n')
            count = 0
            for data in dataloader:
                inputs, reglabels, dotlabels, id_list = data['image'], data['regress'], data['dotted'], data['id']
                inputs = inputs.to(self.device, dtype=torch.float)
                reglabels = reglabels.to(self.device, dtype=torch.float)                
                
                outputs = self.network(inputs)
                loss = criterion(outputs, reglabels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                    
                train_loss+= loss.item()
                
            
            
            print('..On-the-fly augmentation begins...\n')
            rott = self.rotationSet
            np.random.shuffle(rott)
            for ro in np.arange(self.numRotation):
                traindataobj = NeuronDataSet(root_path=self.root_path, dataDir=self.train_dataDir,\
                                              regressLabel=self.train_regressLabel , dottedLabel=self.train_dottedLabel,\
                                                  regStatus=True, dotStatus=True, augStatus=True, \
                                                      num_batch=self.defaultBatchSize, \
                                                          rotationAng=rott[ro])                
                
                dataloader = DataLoader(traindataobj, batch_size=self.defaultBatchSize, \
                                        shuffle=True) #0: cpu
                for data in dataloader:
                    inputs, reglabels, dotlabels, id_list = data['image'], data['regress'], data['dotted'], data['id']
                    inputs = inputs.to(self.device, dtype=torch.float)
                    reglabels = reglabels.to(self.device, dtype=torch.float)                    
                    outputs = self.network(inputs)
                    loss = criterion(outputs, reglabels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                    
                    train_loss+= loss.item()
            
            print('___ Out of rotation loop_\n')                   
                        
            train_loss /= lenData*(self.numRotation+1)
            eval_loss, eval_tp, eval_fn, eval_fp, len_eval = self.validationNetwork()  #VALIDATION          
            eval_loss = eval_loss/len_eval
            eval_recall = eval_tp/(eval_tp+eval_fn+0.00001)
            eval_precision = eval_tp/(eval_tp+eval_fp+0.00001)
            eval_f1 = 2*eval_tp/(2*eval_tp+eval_fn+eval_fp+0.00001)  
            
            
            print('__Capacity calculation__\n')
            traindataobj = NeuronDataSet(root_path=self.root_path, dataDir=self.train_dataDir, \
                                         regressLabel=self.train_regressLabel, dottedLabel=self.train_dottedLabel,\
                                             regStatus=True, dotStatus=True, augStatus=False,\
                                                 num_batch=1, rotationAng=0)  #here dot_list is included in the object
            
            est_cap, retention_cap, error_cap = 0.0, 0.0, 0.0
            # est_cap, retention_cap, error_cap = self.get_capacity(traindataobj)
            est_cap/=lenData
            retention_cap/=lenData
            error_cap/=lenData        
            
            toc = time.time()
            train_loss_hist.append(train_loss)
            eval_loss_hist.append(eval_loss)
            
            if self.manual_count_available:
                expected_precision = (eval_tp)/(eval_tp+self.true_unlabeled_cell+0.00001)
                expected_f1 = 2*eval_tp/(eval_tp +self.true_labeled_cell + self.true_unlabeled_cell)
            else:
                 expected_precision = 0.0 
                 expected_f1 = 0.0
            
            print([best_model['precision'], best_model['recall'], best_model['F1']])
            result_dir = self.write_path +'/' +self.network.name+'_'+self.result_dir_name  #check
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            if eval_recall > best_model['recall']:              
                torch.save(self.network.state_dict(), os.path.join(result_dir,\
                                                                self.network.name+'_'+ self.training_phase_indx \
                                                                    +"_best_weights.pt")) #check
                best_model = {'F1': eval_f1, 'recall': eval_recall, 'precision': eval_precision}
            
            print('epoch {} {:.2f}s. train loss: {:.5f} eval loss: {:.5f}\n.'.format(epoch + 1, toc-tic, train_loss, eval_loss))
            print('eval recall: {:.5f}  eval precision: {:.5f} eval f1: {:.5f}\n.'.format(eval_recall, eval_precision, eval_f1))
            print('Est_cap: {:.5f} Ret_cap: {:.5f} Er_cap: {:.5f}\n.'.format(est_cap, retention_cap, error_cap))
            print('Exp_precision: {:.5f}  Exp_f1: {:.5f}'.format(expected_precision, expected_f1))
            self.write_train_log(eval_f1, epoch, eval_recall, eval_precision, \
                                 est_cap, retention_cap, error_cap, expected_precision, expected_f1 ,self.write_path)
            
            
            # UPDATE newly detected cells
            if self.update_status:
                if epoch in epoch_update_array:  
                    print('Data update step begins...\n')
                    dataloader = DataLoader(traindataobj, batch_size=1, shuffle=False, \
                                    num_workers=self.data_loading_workers_num)
                    
                    countt=0
                    for data in dataloader:
                        inputs, reglabels, dotlabels, id_list = data['image'], data['regress'], data['dotted'], data['id']
                        inputs = inputs.to(self.device, dtype=torch.float)
                        outputs = self.network(inputs)                    
                        outputs = outputs.to('cpu')
                        outputs = np.squeeze(outputs.detach().numpy(), axis=1)  
                        outputs = np.squeeze(outputs,axis=0)
                        center_list = self.getCenterList(outputs)      #function                   
                        
                        
                        # center_list = np.array(center_list)              
                        reglabels = np.squeeze(reglabels, axis=1) 
                        reglabels = np.squeeze(reglabels, axis=0)
                        # print(len(center_list))
                        # print('\n')
                        center_list = self.cellLocationRefinement(center_list, id_list) #center_list os float64
                        # print(len(center_list))
                        # print('\n')                        
                        
                        if len(center_list)>0:
                            reglabels = self.img_coding_function(reglabels,center_list,self.img_coding_scheme)  #Undefined
                            org_idx = id_list[0].split('.')[0]
                            filename = open(os.path.join(self.root_path, self.train_dottedLabel, org_idx+'.txt'),'w')
                            np.savetxt(filename,center_list.astype(int),fmt='%i', delimiter=" ")    #UPDATE step
                            filename.close()
                            cv2.imwrite(os.path.join(self.root_path, self.train_regressLabel,org_idx+'.tif'),reglabels) #UPDATE 
                            
                        if (countt%500)==0:
                            print(countt)
                            print('\n')
                            
                        countt+=1
                        
                    
                    print('Data update step ends...\n')
                print('\n\n')
                    
                    
                    
        torch.cuda.empty_cache()
        return best_model['recall'], best_model['precision'], best_model['F1']
            
            
                     
                    
        
    
    def validationNetwork(self):
        valdataobj = NeuronDataSet(root_path=self.root_path, dataDir=self.val_dataDir,\
                                   regressLabel=self.val_regressLabel, dottedLabel=self.val_dottedLabel, \
                                       regStatus=True, dotStatus=True, augStatus=False, num_batch=1, \
                                           rotationAng=0)
        
        dataloader = DataLoader(valdataobj, batch_size=1, shuffle=False)
            
        val_loss = 0
        acc_tp = 0
        acc_fn = 0
        acc_fp = 0
        count = 0
        criterion = nn.MSELoss()
        self.network.to(self.device)         
        for data in dataloader:
            inputs, reglabels, dotlabels, id_list = data['image'], data['regress'], data['dotted'], data['id']
            inputs = inputs.to(self.device, dtype=torch.float)
            reglabels = reglabels.to(self.device, dtype=torch.float)
            outputs = self.network(inputs)
            loss = criterion(outputs, reglabels)
            val_loss += loss.item()
            
            outputs = outputs.to('cpu')
            outputs = np.squeeze(outputs.detach().numpy(), axis=1)  
            outputs = np.squeeze(outputs,axis=0)
            inputs = inputs.to('cpu').numpy()
            inputs = np.squeeze(inputs, axis=1) 
            inputs = np.squeeze(inputs, axis=0) 
            
            center_list = self.getCenterList(outputs)      #function         
            center_list = np.array(center_list)
            dotlabels = dotlabels.to('cpu')
            dotlabels = np.squeeze(dotlabels.detach().numpy(),axis=0)
            # print(dotlabels)
            # print(center_list)
            cur_tp,cur_fn,cur_fp = self.getStatistics(dotlabels,center_list)  #function
            
            acc_tp += cur_tp
            acc_fn += cur_fn
            acc_fp += cur_fp 
            count+=1
            
            
        return val_loss, acc_tp, acc_fn, acc_fp, count
             
        
        
    def testNetwork(self):
        testdataobj = NeuronDataSet(root_path=self.root_path, dataDir=self.test_dataDir,\
                                   regressLabel=self.test_regressLabel, dottedLabel=self.test_dottedLabel, \
                                       regStatus=False, dotStatus=False, augStatus=False,\
                                           num_batch=1, rotationAng=0)
        
        dataloader = DataLoader(testdataobj, batch_size=1, \
                                shuffle=False)
            
        if self.manual_test_path_status:
             result_dir = self.manual_test_path
        else:
             result_dir = self.write_path +'/' +self.network.name+'_'+self.result_dir_name+'/'+self.training_phase_indx #check
        
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        if self.which_net_to_use == 'Best':
            if self.best_test_weight_name_status:
                best_weights = os.path.join('Cellcounter',self.best_test_weight_name_if_changed+".pt")
            else:
                best_weights = os.path.join(self.write_path +'/' +self.network.name+'_'+self.result_dir_name,\
                                            self.network.name+'_'+ self.training_phase_indx +"_best_weights.pt")
                
            self.network.load_state_dict(torch.load(best_weights, weights_only=True))
            self.network.eval()
               
        self.network.to(self.device)   
        for data in dataloader:
            inputs, reglabels, dotlabels, id_list = data['image'], data['regress'], data['dotted'], data['id']
            org_idx = id_list[0].split('.')[0]
            
            original_img = cv2.imread(os.path.join(self.root_path,self.test_dataDir, org_idx+'.tif'),0)  #check
            inputs = inputs.to(self.device,dtype=torch.float)            
            outputs = self.network(inputs)        
            outputs = outputs.to('cpu')
            outputs = np.squeeze(outputs.detach().numpy(), axis=1) 
            outputs = np.squeeze(outputs,axis=0)
            center_list = self.getCenterList(outputs)
            center_list = np.array(center_list)
            color_input = self.my_gray2color(original_img)
            xx = np.array([])
            yy = np.array([])
            if center_list.size != 0:
#                print(id_list)
#                print(center_list)
                xx,yy = center_list[:,0], center_list[:,1]           
                center_mask = np.zeros(original_img.shape)
                center_mask[xx.astype(np.int), yy.astype(np.int)] = [1]
                color_input[center_mask.astype(bool)] = [0, 0, 255]
            
            cv2.imwrite(result_dir+'/'+org_idx+'_color.tif',color_input)
            
            
    
    
 
    def write_train_log(self, eval_counting, epoch, eval_recall, eval_precision, est_cap, retention_cap, error_cap, \
                        exp_precision, exp_f1, write_path):
        result_dir = self.write_path +'/' +self.network.name+'_'+self.result_dir_name  #check
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
            
        if epoch == 0:
            with open(os.path.join(result_dir, 'acc_logs.txt'), 'a') as log_out:
                log_out.write('epoch \trecall \tprec \tf1_score \texp_prec \texp_f1 \tEst_cap \tRet_cap \tErr_cap \n')

        with open(os.path.join(result_dir, 'acc_logs.txt'), 'a') as log_out:
            log_out.write('{} \t'.format(epoch))
            # for idx in range(len(eval_recall)):
            log_out.write('{:.4f} \t'.format(eval_recall))
            log_out.write('{:.4f} \t'.format(eval_precision))
            log_out.write('{:.4f} \t'.format(eval_counting))
            log_out.write('{:.4f} \t'.format(exp_precision))
            log_out.write('{:.4f} \t'.format(exp_f1))
            log_out.write('{:.4f} \t'.format(est_cap))
            log_out.write('{:.4f} \t'.format(retention_cap))
            log_out.write('{:.4f}'.format(error_cap))
            log_out.write('\n') 

        
        
        
                    
                    
                
                
            
            
            
                
                
            
                
            
        
        
        
        
    
    
