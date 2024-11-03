# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure, morphology
import pickle
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tqdm import tqdm


path_data = 'C:\\Mitotic Event Detection\\Standard dataset\\osfstorage-archive\\'
en = os.listdir(path_data)
path_label = 'C:\\Mitotic Event Detection\\label\\'
path_resave_image = 'C:\\Mitotic Event Detection\\image for observation\\'

CT_list = []
bbox_list = []
for e in tqdm(range(len(en))):
    
    # print(str(e) + 'image is on process')
    im = cv2.imread(path_data+en[e],-1)
    mu = np.mean(im)
    sd = np.std(im)
    Z_mat = (im - mu)/sd
    
    mito_potent = Z_mat > 8
    clea_mp = morphology.remove_small_objects(mito_potent,min_size = 20)
    
    
    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(im)
    # plt.title('Original image',fontsize = 30)
    # plt.subplot(1,2,2)
    # plt.imshow(clea_mp)
    # plt.title('Area for mitotic event',fontsize = 30)
    
    lab_im = measure.label(clea_mp)
    if e < 10:
        pre = '000'
    if e >=10 and e < 100:
        pre = '00'
    if e >= 100 and e < 1000:
        pre = '0'
    if e >= 1000:
        pre = ''
    # np.save(path_label + pre + str(e),lab_im)
    props = measure.regionprops_table(lab_im,properties = ['area','Centroid'
                                                            ,'bbox','label'])
    c0 = [props['Centroid-0']]
    c1 = [props['Centroid-1']]
    Centroid = np.transpose(np.concatenate((c0,c1),axis = 0))
    CT_list.append(Centroid)
    
    b0 = [props['bbox-0']]
    b1 = [props['bbox-1']]
    b2 = [props['bbox-2']]
    b3 = [props['bbox-3']]
    bbox = np.transpose(np.concatenate((b0,b1,b2,b3),axis = 0))
    bbox_list.append(bbox)
    
    pd_props = pd.DataFrame(props)
    
with open("CT_L", "wb") as fp:   #Pickling
    pickle.dump(CT_list, fp)

with open("BB_L", "wb") as fp:   #Pickling
    pickle.dump(bbox_list, fp)
    

   
"""
Path generation
"""

def fig2np(figure):
    figure.canvas.draw()
    rgba_buf = figure.canvas.buffer_rgba()
    (w,h) = figure.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    return rgba_arr

with open("CT_L", "rb") as fp:   # Unpickling
    CT_list = pickle.load(fp)

with open("BB_L", "rb") as fp:   # Unpickling
    bbox_list = pickle.load(fp)

path_list = []
EUpath_list = []
for f in range(len(CT_list) - 1):    
    pre_CT = CT_list[f]
    cur_CT = CT_list[f+1]
    
    for p in range(len(pre_CT)):
        pre_point = pre_CT[p,:]
        EU_dist = np.sqrt(np.sum(np.power(cur_CT - pre_point,2),axis = 1))
        mn = np.min(EU_dist)
        pos_mn = np.argmin(EU_dist)
        fg_replace = False
        
        path = []
        EU_path = []
        if f == 0: 
            if mn <= 20:
                path = [f,p,pos_mn]
                EU_path = [f,mn]
                path_list.append(path)
                EUpath_list.append(EU_path)
            
        if f > 0:
            if mn <= 20:
                for pl in range(len(path_list)):
                    sub_path = path_list[pl]
                    sub_EUpath = EUpath_list[pl]
                    if sub_path[-1] == p and sub_path[0] + len(sub_path) == f + 2 and not fg_replace:
                        sub_path.append(pos_mn)
                        sub_EUpath.append(mn)
                        path_list[pl] = sub_path
                        EUpath_list[pl] = sub_EUpath
                        fg_replace = True
                        
                if not fg_replace:
                    path = [f,p,pos_mn]
                    EU_path = [f,mn]
                    path_list.append(path)
                    EUpath_list.append(EU_path)

video_path = 'C:\\Mitotic Event Detection\\Video for ground truth\\Seq0\\'                
for p in tqdm(range(len(path_list))):
    
    exam_path = path_list[p]
    just_path = exam_path[1:len(exam_path)]
    ind_jp = 0
    fig, ax = plt.subplots(figsize = (30,20))
    path_temp_pic = 'C:\\Mitotic Event Detection\\Temp_pic\\'
    
    if p < 10:
        pre_e = '000'
    if p >=10 and f < 100:
        pre_e = '00'
    if p >= 100 and f < 1000:
        pre_e = '0'
    if p >= 1000:
        pre_e = ''
        
    vid_name = video_path + 'Path_' + str(pre_e) + str(p) + '.mp4'
    
    for f in range(exam_path[0],exam_path[0] + len(exam_path) - 2):
        
        bbox_f = bbox_list[f]
        enl = os.listdir(path_label)
        
        
        im = cv2.imread(path_data+en[f],-1)
        pos = just_path[ind_jp]
        rect = patches.Rectangle((bbox_f[pos,1], bbox_f[pos,0]), 
                                  bbox_f[pos,3] - bbox_f[pos,1], 
                                  bbox_f[pos,2] - bbox_f[pos,0], linewidth=2, 
                                  edgecolor='r', facecolor='none')
        # print(bbox_f[pos,:])
        if f < 10:
            pre = '000'
        if f >=10 and f < 100:
            pre = '00'
        if f >= 100 and f < 1000:
            pre = '0'
        if f >= 1000:
            pre = ''
        
        path_save_fig = path_temp_pic + 'pic_' + pre + str(f)
        ax.clear()
        ax.imshow(im)
        ax.add_patch(rect)
        plt.xlim(bbox_f[8,1] - 50, bbox_f[8,3] + 50)
        plt.ylim(bbox_f[8,0] - 50, bbox_f[8,2] + 50)
        plt.show()
        # plt.pause(0.2)
        plt.savefig(path_save_fig)
        ind_jp = ind_jp+1
        # print(str(f) + ' of ' + str(exam_path[0] + len(exam_path) - 2))
        
    plt.close()
        
    en_temp_pic = os.listdir(path_temp_pic)
    if len(en_temp_pic) > 3:
        img_seq = []
        
        for i in range(len(en_temp_pic)):
            im = cv2.imread(path_temp_pic + en_temp_pic[i])
            img_seq.append(im)
        
        # print('Length of image series is ' + str(len(img_seq)))
        height,width,layers = img_seq[0].shape
        video = cv2.VideoWriter(vid_name,-1,5,(width,height))
        
        for j in range(len(img_seq)):
            video.write(img_seq[j])
        
        cv2.destroyAllWindows()
        video.release()
        
        for f in range(len(en_temp_pic)):
            os.remove(path_temp_pic + en_temp_pic[f])
            
        print(str(p) + ' out of ' + str(len(path_list)) + ' is processed')
        # print('Path length is ' + str(len(just_path)))




                    


                    
                
        
    
    

    
    
