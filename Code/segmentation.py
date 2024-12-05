# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:48:44 2022

@author: r0814655
"""

import numpy as np
import pandas as pd
import cv2
import os
from skimage.filters import threshold_otsu, threshold_local
import skimage.filters as filters
from skimage import exposure
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage import measure
from skimage import morphology as smh
from skimage import util
from skimage import exposure
from skimage import io



for s in range(1,5):
    # s = 1
    seq = 'Seq' + str(s)
    path_data = 'C:\\Mitotic Event Detection\\Standard dataset\\' + seq + '\\'
    # path_props = 'C:\\Mitotic Event Detection\\Region properties 2800th\\' + seq +'\\'
    # path_results = 'C:\\Mitotic Event Detection\\Segmentation results\\' + seq +'\\'
    # path_label = 'C:\\Mitotic Event Detection\\Label pic 2800th\\' + seq +'\\'
    
    path_props = 'C:\\Mitotic Event Detection\\2500_th\\Region properties\\' + seq +'\\'
    path_results = 'C:\\Mitotic Event Detection\\2500_th\\Segmentation results\\' + seq +'\\'
    path_label = 'C:\\Mitotic Event Detection\\2500_th\\Label pic\\' + seq +'\\'
    
    path_fc = 'C:\\Mitotic Event Detection\\Temp frame capture\\'
    path_seq_vid = 'C:\\Mitotic Event Detection\\Seq vid\\'
    
    path_seq_vid = 'C:\\Mitotic Event Detection\\2500_th\\Seq vid\\'
    
    en = os.listdir(path_data)
    
    for v in range(len(en)):
        im = cv2.imread(path_data + en[v],-1)
        th = threshold_otsu(im)
        binary = im > th
        # binary_def_th = im > 2800
        binary_def_th = im > 2600
        rev_obj_bin = smh.remove_small_objects(binary_def_th, min_size = 50)
        dilated = smh.dilation(rev_obj_bin,footprint = smh.disk(10))
        contours = measure.find_contours(dilated, 0.8)
        
        labs = measure.label(dilated)
        
        
        test_props = measure.regionprops_table(labs,im,properties = ['label','Centroid','bbox'])
        
        pd_props = pd.DataFrame(test_props)
        
        # plt.subplots(figsize = (200,200))
        # plt.subplot(2,2,1)
        # plt.imshow(im)
        # plt.title('Original image',fontsize = 30)
        # plt.subplot(2,2,2)
        # plt.imshow(binary_def_th)
        # plt.title('Threshold image at 2800',fontsize = 30)
        # plt.subplot(2,2,3)
        # plt.imshow(rev_obj_bin)
        # plt.title('Remove small objects',fontsize = 30)
        # plt.subplot(2,2,4)
        # plt.imshow(dilated)
        # plt.title('Dilated',fontsize = 30)
        
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(im, cmap = 'gray')
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=5, color = 'red')
        plt.title('Mitosis area',fontsize = 30)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        if v < 10:
            pre = '000'
        if v >=10 and v < 100:
            pre = '00'
        if v > 100 and v < 1000:
            pre = '0'
        if v > 1000:
            pre = ''
            
        plt.savefig(path_fc + 'con_img_' + pre + str(v) + '.png')
        io.imsave(path_label + 'lab_'+ pre + str(v) + '.tif',labs,check_contrast = False)
        pd_props.to_csv(path_props + 'props_' + pre + str(v) + '.csv',index=None)
        plt.savefig(path_results + 'segmentation_'+ pre + str(v) + '.png')
        plt.close()
        
        plt.close()
                    
    temp_fig = os.listdir(path_fc)
    vid_name = path_seq_vid + 'Vid_' + str(s) + '.mp4'
    img = []
    for t in range(len(temp_fig)):
        img.append(cv2.imread(path_fc + temp_fig[t],-1))
        
    height,width,layers = img[0].shape
    video = cv2.VideoWriter(vid_name,-1,10,(width,height))
    
    for j in range(len(img)):
        video.write(img[j])
    
    cv2.destroyAllWindows()
    video.release()
    
    for f in range(len(temp_fig)):
        os.remove(path_fc + temp_fig[f])
        
    for e in range(len(en)):
        im = cv2.imread(path_data + en[e],-1)
        th = threshold_otsu(im)
        binary = im > th
        # binary_def_th = im > 2800
        binary_def_th = im > 2500
        rev_obj_bin = smh.remove_small_objects(binary_def_th, min_size = 50)
        dilated = smh.dilation(rev_obj_bin,footprint = smh.disk(10))
        contours = measure.find_contours(dilated, 0.8)
        
        labs = measure.label(dilated)
        
        
        test_props = measure.regionprops_table(labs,im,properties = ['label','Centroid','bbox'])
        
        pd_props = pd.DataFrame(test_props)
        
        fig, ax = plt.subplots(figsize=(200,200))
        ax.imshow(im, cmap = 'gray')
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=5)
        plt.title('Mitosis area',fontsize = 30)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        if e < 10:
            pre_save = '000' 
        if e >= 10 and e < 100:
            pre_save = '00'
        if e >= 100 and e < 1000:
            pre_save = '0'
        if e >= 1000:
            pre_save = ''
            
        io.imsave(path_label + 'lab_'+ pre_save + str(e) + '.tif',labs,check_contrast = False)
        pd_props.to_csv(path_props + 'props_' + pre_save + str(e) + '.csv',index=None)
        plt.savefig(path_results + 'segmentation_'+ pre_save + str(e) + '.png')
        plt.close()
        
    print('Sequence ' + str(s) + ' is completed')
    
    
    
    
    
    
