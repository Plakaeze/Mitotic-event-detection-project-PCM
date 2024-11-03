# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:25:15 2022

@author: r0814655
"""

import numpy as np
import pandas as pd
import cv2
import os
from skimage.filters import threshold_otsu, threshold_local
from skimage import exposure
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage import measure
from skimage.io import imsave
from skimage import morphology as smh
import pickle
from tqdm import tqdm


path_data = 'C:\\Mitotic Event Detection\\Standard dataset\\osfstorage-archive\\'
path_save_props = 'C:\\Mitotic Event Detection\\Cells properties\\'
path_save_lab_im = 'C:\\Mitotic Event Detection\\Cell Label\\'

with open("start_position", "rb") as fp:   # Unpickling
    start_position = pickle.load(fp)    

en = os.listdir(path_save_props)

mito_marked = []
eu_marked = []
cls_marked = []

print('Path generations...')
for f in range(len(en) - 1):
    cur_props = pd.read_csv(path_save_props + en[f])
    nxt_props = pd.read_csv(path_save_props + en[f+1])
    
    c0_cur = np.asarray(cur_props['Centroid-0'])
    c1_cur = np.asarray(cur_props['Centroid-1'])
    c0_nxt = np.asarray(nxt_props['Centroid-0'])
    c1_nxt = np.asarray(nxt_props['Centroid-1'])
    cur_label = np.asarray(cur_props['label'])
    nxt_label = np.asarray(nxt_props['label'])
    
    mito_subpath = []
    mito_subeu = []
    mito_sub_cls = []
    
    for e in range(len(c0_cur)):
        pwx = np.power(c0_nxt - c0_cur[e],2)
        pwy = np.power(c1_nxt - c1_cur[e],2)
        
        eu_dist = np.sqrt(pwx + pwy)
        
        mn = np.min(eu_dist)
        mn_idx = np.argmin(eu_dist)
        
        eu_dist[mn_idx] = np.Inf
        clsed = np.min(eu_dist)
        clsed_idx = np.argmin(eu_dist)
        
        if mn < 50:
            mito_subpath.append([cur_label[e],nxt_label[mn_idx]])
            mito_subeu.append([cur_label[e],nxt_label[mn_idx],mn])
            mito_sub_cls.append([cur_label[e],nxt_label[clsed_idx],clsed])
    
    mito_marked.append(mito_subpath)
    eu_marked.append(mito_subeu)
    cls_marked.append(mito_sub_cls)
    
    

full_mito_path = []
full_eu = []
full_cls = []
full_cls_ind = []

for i in tqdm(range(len(mito_marked))):
    cons_path = mito_marked[i]
    eu_path = eu_marked[i]
    cls_path = cls_marked[i]
    
    if i == 0:
        for cp in range(len(cons_path)):
            
            temp_fmp = np.concatenate(([i],cons_path[cp],[i+1]),axis = 0)
            temp_eu = np.concatenate(([i],[eu_path[cp][2]],[i+1]),axis = 0)
            temp_cls = np.concatenate(([i],[cls_path[cp][2]],[i+1]),axis = 0)
            temp_cls_ind = np.concatenate(([i],cls_path[cp][0:2],[i+1]),axis = 0)
            
            full_mito_path.append(temp_fmp)
            full_eu.append(temp_eu)
            full_cls.append(temp_cls)
            full_cls_ind.append(temp_cls_ind)
    
    if i > 0:
        keep_con_idx = []
        old_len = len(full_mito_path)
        for smpc in range(old_len):
            old_mp = full_mito_path[smpc]
            old_eu = full_eu[smpc]
            old_cls = full_cls[smpc]
            old_cls_ind = full_cls_ind[smpc]
            for ssmpc in range(len(cons_path)):
                
                if old_mp[-2] == cons_path[ssmpc][0] and old_mp[-1] == i:
                    
                    temp_fmp = np.concatenate((old_mp[0:-1],[cons_path[ssmpc][1]],[i+1]),axis = 0)
                    temp_eu = np.concatenate((old_eu[0:-1],[eu_path[ssmpc][2]],[i+1]),axis = 0)
                    temp_cls = np.concatenate((old_cls[0:-1],[cls_path[ssmpc][2]],[i+1]),axis = 0)
                    temp_cls_ind = np.concatenate((old_cls_ind[0:-1],[cls_path[ssmpc][1]],[i+1]),axis = 0)
                    
                    
                    full_mito_path[smpc] = temp_fmp
                    full_eu[smpc] = temp_eu
                    full_cls[smpc] = temp_cls
                    full_cls_ind[smpc] = temp_cls_ind
                    
                    keep_con_idx.append(ssmpc)
            
        for suc in range(len(cons_path)):
            if not suc in keep_con_idx:
                
                temp_fmp = np.concatenate(([i],cons_path[suc],[i+1]),axis = 0)
                temp_eu = np.concatenate(([i],[eu_path[suc][2]],[i+1]),axis = 0)
                temp_cls = np.concatenate(([i],[cls_path[suc][2]],[i+1]),axis = 0)
                temp_cls_ind = np.concatenate(([i],cls_path[suc][0:2],[i+1]),axis = 0)
                
                full_mito_path.append(temp_fmp)
                full_eu.append(temp_eu)
                full_cls.append(temp_cls)
                full_cls_ind.append(temp_cls_ind)
                
    # print('Path ' + str(i) + ' of ' + str(len(mito_marked)) + ' is completed.')

# Synce full mito path

synce_fmp = []
synce_cls = []
synce_eu = []
synce_cls_ind = []
for sp in range(len(start_position)):
    
    startframe = start_position[sp][0]
    start_cell = start_position[sp][1]
    
    for fmp in range(len(full_mito_path)):
        temp_path = full_mito_path[fmp]
        temp_cls = full_cls[fmp]
        temp_cls_ind = full_cls_ind[fmp]
        temp_eu = full_eu[fmp]
        
        if temp_path[0] == startframe and temp_path[1] == start_cell:
            synce_fmp.append(temp_path)
            synce_cls.append(temp_cls)
            synce_eu.append(temp_eu)
            synce_cls_ind.append(temp_cls_ind)
                
cons_mito_path = []
cons_mito_eu = []
cons_mito_cls = []
cons_mito_cls_ind = []


for m in range(len(synce_fmp)):
    
    tar_path = synce_fmp[m]
    tar_path = tar_path[0:-1]
    tar_eu = synce_eu[m]
    tar_eu = tar_eu[0:-1]
    tar_cls = synce_cls[m]
    tar_cls = tar_cls[0:-1]
    tar_cls_ind = synce_cls_ind[m]
    tar_cls_ind = tar_cls_ind[0:-1]
    
    path_len = len(tar_path)
    frame_start = tar_path[0]
    early_z = np.zeros(frame_start)
    post_z = np.zeros(len(en) - (len(tar_path) - 1) - frame_start)
    
    processed_path = np.concatenate((early_z,tar_path[1:len(tar_path)],post_z),axis = 0)
    processed_eu = np.concatenate((early_z,tar_eu[1:len(tar_path)],post_z),axis = 0)
    processed_cls = np.concatenate((early_z,tar_cls[1:len(tar_path)],post_z),axis = 0)
    processed_cls_ind = np.concatenate((early_z,tar_cls_ind[1:len(tar_cls_ind)],post_z),axis = 0)
    
    if path_len < 15 and path_len > 3:
        cons_mito_path.append(processed_path)
        cons_mito_eu.append(processed_eu)
        cons_mito_cls.append(processed_cls)
        cons_mito_cls_ind.append(processed_cls_ind)
        

# Similarity calculation
print('Similarity calculation')

np_cons_mito_path = np.asarray(cons_mito_path)
rekeep_cons_mito_path = []

np_cons_mito_eu = np.asarray(cons_mito_eu)
rekeep_cons_mito_eu = []

np_cons_mito_cls = np.asarray(cons_mito_cls)
rekeep_cons_mito_cls = []

np_cons_mito_cls_ind = np.asarray(cons_mito_cls_ind)
rekeep_cons_mito_cls_ind = []

skip_fg = False
kept_ind = []
sim_list = []
pos_range = []
for f in range(len(cons_mito_path[0]) - 1):
    for j in range(len(cons_mito_path) - 1):
        if not skip_fg:
            if np_cons_mito_path[j,f] != 0 and np_cons_mito_path[j,f] != np_cons_mito_path[j+1,f]:
                bef = np_cons_mito_path[j,:]
                af = np_cons_mito_path[j+1,:]
                score = 0
                div = 0
                k = f
                while bef[k] != 0 and k <= len(bef) - 2:
                    div = div + 1
                    if bef[k] == af[k]:
                        score = score + 1
                    k = k+1
                    
                similarity = score/div
                sim_list.append([score,div,similarity,f,j,j+1])
                
                if similarity <  0.4 and not j in kept_ind:
                    pos_range.append([f,f+div])
                    rekeep_cons_mito_path.append(cons_mito_path[j])
                    rekeep_cons_mito_eu.append(cons_mito_eu[j])
                    rekeep_cons_mito_cls.append(cons_mito_cls[j])
                    rekeep_cons_mito_cls_ind.append(cons_mito_cls_ind[j])
                    skip_fg = True
                    kept_ind.append(j)
                    
        if skip_fg:
            skip_fg = False

# Verify the path of mitosis
en_lab = os.listdir(path_save_lab_im)
en_im = os.listdir(path_data)
path_temp_fig = 'C:\\Mitotic Event Detection\\Temp_pic\\'
path_save_vid = 'C:\\Mitotic Event Detection\\Video for ground truth\\'
path_sequence_array = 'C:\\Mitotic Event Detection\\Sequence array\\'
for rp in range(len(rekeep_cons_mito_path)): 

    view_path = rekeep_cons_mito_path[rp]
    pos_view = pos_range[rp]
    frame_range = list(range(pos_view[0],pos_view[1]))
    cap_view_path = view_path[view_path != 0]
    os.makedirs(path_sequence_array + 'Sequence_' + str(rp))
    path_save_array = path_sequence_array + 'Sequence_' + str(rp) + '//'
    for v in range(len(cap_view_path)):
        lab_im = cv2.imread(path_save_lab_im + en_lab[frame_range[v]], -1)
        im = cv2.imread(path_data + en_im[frame_range[v]],-1)
        props = pd.read_csv(path_save_props + en[frame_range[v]])
        min_x = props['bbox-1'][cap_view_path[v]-1]
        max_x = props['bbox-3'][cap_view_path[v]-1]
        min_y = props['bbox-0'][cap_view_path[v]-1]
        max_y = props['bbox-2'][cap_view_path[v]-1]
        c_im = np.zeros(lab_im.shape)
        c_im[lab_im != cap_view_path[v]] = 0
        c_im[lab_im == cap_view_path[v]] = 255
        contours = measure.find_contours(c_im)
        
        # fig, ax = plt.subplots()
        # ax.imshow(im, cmap='gray')
            
        # for contour in contours:
        #     ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
        
        # ax.axis('image')
        # ax.set_xticks([])
        # ax.set_yticks([])
        
        # plt.xlim((min_x - 100,max_x + 100))
        # plt.ylim((min_y - 100,max_y + 100))
    
        # plt.show()
        
        if v < 10:
            pre = '000'
        if v >= 10 and v < 100:
            pre = '00'
        if v >=100 and v < 1000:
            pre = '0'
        if v > 1000:
            pre = ''
        # plt.savefig(path_temp_fig + 'con_img_' + pre + str(v) + '.png')
        # plt.close()
        
        detected_area = im[min_y - 50:max_y + 50,min_x - 50:max_x + 50]
        np.save(path_save_array + 'array_' + pre + str(v), detected_area)
        
    
    # temp_fig = os.listdir(path_temp_fig)
    # if rp < 10:
    #     pre_vid = '000'
    # if rp >= 10 and rp < 100:
    #     pre_vid = '00'
    # if rp >=100 and rp < 1000:
    #     pre_vid = '0'
    # if rp > 1000:
    #     pre_vid = ''
    # vid_name = path_save_vid + 'Vid_' + pre + str(rp) + '.mp4'
    # img = []
    # for t in range(len(temp_fig)):
    #     img.append(cv2.imread(path_temp_fig + temp_fig[t],-1))
        
    # height,width,layers = img[0].shape
    # video = cv2.VideoWriter(vid_name,-1,1,(width,height))
    
    # for j in range(len(img)):
    #     video.write(img[j])
    
    # cv2.destroyAllWindows()
    # video.release()
    
    # for f in range(len(temp_fig)):
    #     os.remove(path_temp_fig + temp_fig[f])