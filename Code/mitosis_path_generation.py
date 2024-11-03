# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:55:39 2022

@author: r0814655
"""

import os
import pandas as pd
import numpy as np
import pickle
import cv2
from skimage import measure
from skimage import io
import matplotlib.pyplot as plt

for s in range(1,5):
    seq = 'Seq' + str(s)    
    path_area_props = 'C:\\Mitotic Event Detection\\Region properties\\' + seq + '\\'
    path_data = 'C:\\Mitotic Event Detection\\Standard dataset\\' + seq +'\\'
    
    # path_area_props = 'C:\\Mitotic Event Detection\\2500_th\\Region properties\\' + seq + '\\'
    # path_data = 'C:\\Mitotic Event Detection\\Standard dataset\\' + seq +'\\'
    
    
    en_area = os.listdir(path_area_props)
    en_data = os.listdir(path_data)
    
    # Area path
    mito_marked = []
    eu_marked = []
    
    
    for f in range(len(en_area) - 1):
        cur_props = pd.read_csv(path_area_props + en_area[f])
        nxt_props = pd.read_csv(path_area_props + en_area[f+1])
        
        c0_cur = np.asarray(cur_props['Centroid-0'])
        c1_cur = np.asarray(cur_props['Centroid-1'])
        c0_nxt = np.asarray(nxt_props['Centroid-0'])
        c1_nxt = np.asarray(nxt_props['Centroid-1'])
        cur_label = np.asarray(cur_props['label'])
        nxt_label = np.asarray(nxt_props['label'])
        
        mito_subpath = []
        mito_subeu = []
        
        
        for e in range(len(c0_cur)):
            pwx = np.power(c0_nxt - c0_cur[e],2)
            pwy = np.power(c1_nxt - c1_cur[e],2)
            
            eu_dist = np.sqrt(pwx + pwy)
            
            mn = np.min(eu_dist)
            mn_idx = np.argmin(eu_dist)
            
            
            if mn < 50:
                mito_subpath.append([cur_label[e],nxt_label[mn_idx]])
                mito_subeu.append([cur_label[e],nxt_label[mn_idx],mn])
                
        
        mito_marked.append(mito_subpath)
        eu_marked.append(mito_subeu)
        
        
        
    full_mito_path = []
    full_eu = []
    
    
    for i in range(len(mito_marked)):
        cons_path = mito_marked[i]
        eu_path = eu_marked[i]
        
        if i == 0:
            for cp in range(len(cons_path)):
                
                temp_fmp = np.concatenate(([i],cons_path[cp],[i+1]),axis = 0)
                temp_eu = np.concatenate(([i],[eu_path[cp][2]],[i+1]),axis = 0)
                
                full_mito_path.append(temp_fmp)
                full_eu.append(temp_eu)
    
        
        if i > 0:
            keep_con_idx = []
            old_len = len(full_mito_path)
            for smpc in range(old_len):
                old_mp = full_mito_path[smpc]
                old_eu = full_eu[smpc]
                for ssmpc in range(len(cons_path)):
                    
                    if old_mp[-2] == cons_path[ssmpc][0] and old_mp[-1] == i:
                        
                        temp_fmp = np.concatenate((old_mp[0:-1],[cons_path[ssmpc][1]],[i+1]),axis = 0)
                        temp_eu = np.concatenate((old_eu[0:-1],[eu_path[ssmpc][2]],[i+1]),axis = 0)
                        
                        
                        full_mito_path[smpc] = temp_fmp
                        full_eu[smpc] = temp_eu
                        
                        keep_con_idx.append(ssmpc)
                
            for suc in range(len(cons_path)):
                if not suc in keep_con_idx:
                    
                    temp_fmp = np.concatenate(([i],cons_path[suc],[i+1]),axis = 0)
                    temp_eu = np.concatenate(([i],[eu_path[suc][2]],[i+1]),axis = 0)
                    
                    full_mito_path.append(temp_fmp)
                    full_eu.append(temp_eu)
    
    filtered_full_mito_path = []
    for f in range(len(full_mito_path)):
        temp_cons_path = full_mito_path[f]
        if len(temp_cons_path) < 30 and len(temp_cons_path) > 3:
            filtered_full_mito_path.append(temp_cons_path)
    
    # path_path = 'C:\\Mitotic Event Detection\\Full mito path\\'+ seq + '\\full_mito_path'
    # path_path = 'C:\\Mitotic Event Detection\\2500_th\\Full mito path\\'+ seq + '\\full_mito_path'
    path_path = 'C:\\Mitotic Event Detection\\Full mito path longer\\'+ seq + '\\full_mito_path'
    
    with open(path_path, "wb") as fp:   # Pickling
        pickle.dump(filtered_full_mito_path, fp)
        
    cons_mito_path = []
    cons_mito_eu = []
    pos_range = []
    
    for m in range(len(filtered_full_mito_path)):
        
        tar_path = filtered_full_mito_path[m]
        pos_range.append([tar_path[0],tar_path[-1] + 1])
        tar_path = tar_path[0:-1]
        tar_eu = full_eu[m]
        tar_eu = tar_eu[0:-1]
    
        
        path_len = len(tar_path)
        frame_start = tar_path[0]
        early_z = np.zeros(frame_start)
        post_z = np.zeros(len(en_data) - (len(tar_path) - 1) - frame_start)
        
        processed_path = np.concatenate((early_z,tar_path[1:len(tar_path)],post_z),axis = 0)
        processed_eu = np.concatenate((early_z,tar_eu[1:len(tar_path)],post_z),axis = 0)
    
        cons_mito_path.append(processed_path)
        cons_mito_eu.append(processed_eu)
    
    path_lab = 'C:\\Mitotic Event Detection\\Label pic\\' + seq +'\\'
    path_temp_fig = 'C:\\Mitotic Event Detection\\Temp_pic\\'
    # path_save_vid = 'C:\\Mitotic Event Detection\\Video for ground truth\\' + seq + '\\'
    path_save_vid = 'C:\\Mitotic Event Detection\\Video for ground truth longer\\' + seq + '\\'
    
    # path_lab = 'C:\\Mitotic Event Detection\\2500_th\\Label pic\\' + seq +'\\'
    # path_temp_fig = 'C:\\Mitotic Event Detection\\2500_th\\Temp_pic\\'
    # path_save_vid = 'C:\\Mitotic Event Detection\\2500_th\\Video for ground truth\\' + seq + '\\'
    
    en_lab = os.listdir(path_lab)
    
    for rp in range(len(cons_mito_path)): 
        view_path = cons_mito_path[rp]
        pos_view = pos_range[rp]
        frame_range = list(range(pos_view[0],pos_view[1]+1))
        cap_view_path = view_path[view_path != 0]
        
        for v in range(len(cap_view_path)):
            lab_im = cv2.imread(path_lab + en_lab[frame_range[v]], -1)
            im = cv2.imread(path_data + en_data[frame_range[v]],-1)
            props = pd.read_csv(path_area_props + en_area[frame_range[v]])
            min_x = props['bbox-1'][cap_view_path[v]-1]
            max_x = props['bbox-3'][cap_view_path[v]-1]
            min_y = props['bbox-0'][cap_view_path[v]-1]
            max_y = props['bbox-2'][cap_view_path[v]-1]
            c_im = lab_im
            c_im[lab_im != cap_view_path[v]] = 0
            c_im[lab_im == cap_view_path[v]] = 255
            contours = measure.find_contours(c_im)
            
            fig, ax = plt.subplots()
            ax.imshow(im, cmap='gray')
                
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
            
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.xlim((min_x - 50,max_x + 50))
            plt.ylim((min_y - 50,max_y + 50))
        
            plt.show()
            
            if v < 10:
                pre = '00'
            if v >=10 and v < 100:
                pre = '0'
            if v > 100:
                pre = ''
            plt.savefig(path_temp_fig + 'con_img_' + pre + str(v) + '.png')
            plt.close()
            
        
        temp_fig = os.listdir(path_temp_fig)
        if rp < 10:
            pre_vid = '00'
        if rp >=10 and v < 100:
            pre_vid = '0'
        if rp > 100:
            pre_vid = ''
        vid_name = path_save_vid + 'Vid_' + pre_vid + str(rp) + '.mp4'
        img = []
        for t in range(len(temp_fig)):
            img.append(cv2.imread(path_temp_fig + temp_fig[t],-1))
            
        height,width,layers = img[0].shape
        video = cv2.VideoWriter(vid_name,-1,1,(width,height))
        
        for j in range(len(img)):
            video.write(img[j])
        
        cv2.destroyAllWindows()
        video.release()
        
        for f in range(len(temp_fig)):
            os.remove(path_temp_fig + temp_fig[f])
            

rp = 154            
view_path = cons_mito_path[rp]
pos_view = pos_range[rp]
frame_range = list(range(pos_view[0],pos_view[1]+1))
cap_view_path = view_path[view_path != 0]
fig, ax = plt.subplots(1, len(frame_range) - 1, sharex=True)
fig.subplots_adjust(hspace=0)
for v in range(len(cap_view_path)):
    lab_im = cv2.imread(path_lab + en_lab[frame_range[v]], -1)
    im = cv2.imread(path_data + en_data[frame_range[v]],-1)
    props = pd.read_csv(path_area_props + en_area[frame_range[v]])
    min_x = props['bbox-1'][cap_view_path[v]-1]
    max_x = props['bbox-3'][cap_view_path[v]-1]
    min_y = props['bbox-0'][cap_view_path[v]-1]
    max_y = props['bbox-2'][cap_view_path[v]-1]
    c_im = lab_im
    c_im[lab_im != cap_view_path[v]] = 0
    c_im[lab_im == cap_view_path[v]] = 255
    contours = measure.find_contours(c_im)
    
    
    ax[v].imshow(im, cmap='gray')
    
    for contour in contours:
        ax[v].plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
    
    ax[v].axis('image')
    ax[v].set_xticks([])
    ax[v].set_yticks([])
    ax[v].set_xlim((min_x - 50,max_x + 50))
    ax[v].set_ylim((min_y - 50,max_y + 50))
    
    