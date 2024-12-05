# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:36:42 2022

@author: r0814655
"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import util
from skimage import io
from skimage import morphology as smh
from skimage import exposure
from skimage.filters import threshold_local
from skimage.filters import threshold_li
from skimage import measure
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu, threshold_isodata
from skimage.feature import hog
import pickle

# seq = 'Seq' + str(s)
seq = 'Seq1'
path_area_props = 'C:\\Mitotic Event Detection\\Region properties 2800th\\' + seq +'\\'
path_lab = 'C:\\Mitotic Event Detection\\Label pic 2800th\\' + seq +'\\'
path_data = 'C:\\Mitotic Event Detection\\Standard dataset\\' + seq +'\\'
path_save_mitosis_results = 'C:\\Mitotic Event Detection\\New detected mitotic frame\\' + seq +'\\'
path_path = 'C:\\Mitotic Event Detection\\Full mito path\\'+ seq + '\\full_mito_path 2800th'


en_data = os.listdir(path_data)
en_props = os.listdir(path_area_props)
en_labs = os.listdir(path_lab)

with open(path_path, "rb") as fp:   # Unpickling
    full_mito_path = pickle.load(fp)

# for p in range(len(full_mito_path)):
# p = 24
# test_path = full_mito_path[p]
# start_frame = test_path[0]
# stop_frame = test_path[-1]
# path = test_path[1:-1]
# path_ind = 0

# distance_factors = []
# phe_area_ecc = []
# obs_eu = []
# mu_inten_path = []
# max_inten_path = []

# for f in range(start_frame,stop_frame+1):
#     props = pd.read_csv(path_area_props + en_props[f])
#     lab = cv2.imread(path_lab+en_labs[f],-1)
#     im = cv2.imread(path_data + en_data[f],-1)
#     c0_area = np.asarray(props['Centroid-0'])[path[path_ind]-1]
#     c1_area = np.asarray(props['Centroid-1'])[path[path_ind]-1]
    
#     marker = np.zeros(lab.shape,dtype=bool)
#     marker[lab == path[path_ind]] = True
    
#     invert = util.invert(im)
#     coordinates = peak_local_max(invert, min_distance=5, labels = marker)
    
#     c0_min = coordinates[:,0]
#     c1_min = coordinates[:,1]
    
#     pwx = np.power(c0_area - c0_min,2)
#     pwy = np.power(c1_area - c1_min,2)
#     eu = np.sqrt(pwx+pwy)
    
#     mn_eu1 = np.min(eu)
#     idx_mn1 = np.argmin(eu)
#     eu[idx_mn1] = np.Inf
#     mn_eu2 = np.min(eu)
#     idx_mn2 = np.argmin(eu)
#     obs_eu.append([mn_eu1,mn_eu2])
    
#     dis_fac = mn_eu1/mn_eu2
#     distance_factors.append(dis_fac)
    
#     off = 10
#     min_x = props['bbox-1'][path[path_ind]-1] - off
#     max_x = props['bbox-3'][path[path_ind]-1] + off
#     min_y = props['bbox-0'][path[path_ind]-1] - off
#     max_y = props['bbox-2'][path[path_ind]-1] + off
    
#     if props['bbox-1'][path[path_ind]-1] - off < 0:
#         min_x = props['bbox-1'][path[path_ind]-1]
#     if props['bbox-3'][path[path_ind]-1] + off > im.shape[1]:
#         max_x = props['bbox-3'][path[path_ind]-1]
#     if props['bbox-0'][path[path_ind]-1] - off < 0:
#         min_y = props['bbox-0'][path[path_ind]-1]
#     if props['bbox-2'][path[path_ind]-1] + off > im.shape[0]:
#         max_y = props['bbox-2'][path[path_ind]-1]
    
#     cropped_im = im[min_y:max_y,min_x:max_x]
#     cropped_marker = marker[min_y:max_y,min_x:max_x]
#     opt_th_for_phe_im = threshold_li(cropped_im)
#     # area_phe_im = cropped_im > opt_th_for_phe_im
#     # area_phe_im = cropped_im > 2800
    
#     p2_im, p98_im = np.percentile(cropped_im, (2, 95))
#     cropped__cs_im = exposure.rescale_intensity(cropped_im, in_range=(p2_im, p98_im))
    
#     opt_th_for_phe = threshold_isodata(cropped__cs_im)
#     area_phe_im = cropped__cs_im > opt_th_for_phe
    
#     lab_cim = measure.label(area_phe_im)
#     props_cim = measure.regionprops_table(lab_cim,cropped_im,properties = 
#                                                           ['label','area','intensity_mean',
#                                                           'axis_major_length','axis_minor_length',
#                                                           'eccentricity','intensity_min',
#                                                           'intensity_max','coords','centroid','bbox'])
#     cim_phe_c0 = props_cim['centroid-0']
#     cim_phe_c1 = props_cim['centroid-1']
    
#     cim_eu = np.sqrt(np.power(c0_area - cim_phe_c0,2) + np.power(c1_area - cim_phe_c1,2))
#     phe_area_ecc.append(props_cim['eccentricity'][np.argmin(cim_eu)])
#     mu_inten_path.append(props_cim['intensity_mean'][np.argmin(cim_eu)])
#     max_inten_path.append(np.max(cropped_im))
    
#     path_ind = path_ind+1
    
# ecc_factors = np.asarray(phe_area_ecc)
# dist_factor = np.asarray(distance_factors)
# arr_max_inten = np.asarray(max_inten_path)
# arr_mean_white_area = np.asarray(mu_inten_path)
# intensity_factor = arr_mean_white_area/np.max(arr_max_inten)
# arr_combined_factors = np.transpose(np.asarray([distance_factors,intensity_factor,ecc_factors]))

# mitosis_factor = 0.1*dist_factor + 0.8*intensity_factor + 0.1*ecc_factors
# mitosis_frame = start_frame + np.argmax(mitosis_factor)
# mito_labels = path[np.argmax(mitosis_factor)]
# mito_props = pd.read_csv(path_area_props + en_props[mitosis_frame])
# mito_labs = cv2.imread(path_lab+en_labs[mitosis_frame],-1)
# mito_im = cv2.imread(path_data + en_data[mitosis_frame],-1)


# mito_c0_area = np.asarray(mito_props['Centroid-0'])[mito_labels-1]
# mito_c1_area = np.asarray(mito_props['Centroid-1'])[mito_labels-1]


# mito_invert = util.invert(mito_im)
# mito_marker = np.zeros(mito_labs.shape,dtype=bool)
# mito_marker[mito_labs == mito_labels] = True

# mito_coordinates = peak_local_max(mito_invert, min_distance=5, labels = mito_marker)


# fig, ax = plt.subplots(figsize=(20,50))
# ax.imshow(mito_im, cmap = 'gray')
# plt.xlim((int(mito_c1_area - 50),int(mito_c1_area + 50)))
# plt.ylim((int(mito_c0_area - 50),int(mito_c0_area+50)))
# ax.plot(mito_coordinates[:, 1], mito_coordinates[:, 0], 'r.',markersize = 20)
# ax.plot(mito_c1_area, mito_c0_area, 'b.',markersize = 20)
    
# if p < 10:
#     pre_vid = '00'
# if p >=10 and p < 100:
#     pre_vid = '0'
# if p > 100:
#     pre_vid = ''
    
# plt.savefig(path_save_mitosis_results + pre_vid + str(p) + '.png')
# plt.close()



column_features = ['Area of mother cell','Major axis of mother cell','Minor axis of mother cell',
                    'Circularity of mother cell', 'Mean intensity of mother cell',
                    'Max intensity of mother cell','Min intensity of mother cell','Axis ratio of mother cell',
                    'Area of daughter cell','Major axis of daughter cell','Minor axis of daughter cell',
                    'Circularity of daughter cell', 'Mean intensity of daughter cell',
                    'Max intensity of daughter cell','Min intensity of daughter cell','Axis ratio of daughter cell',
                    'Area ratio', 'Major axis ratio', 'Minor axis ratio', 'Circularity ratio','Mean intensity ratio',
                    'Max intensity ratio','Min intensity ratio','Axis ratio ratio',
                    'Area diff', 'Major axis diff', 'Minor axis diff', 'Circularity diff','Mean intensity diff',
                    'Max intensity diff','Min intensity diff','Axis ratio diff',
                    'Area of phenomenon','Major axis of phenomenon','Minor axis of phenomenon',
                    'Circularity of phenomenon', 'Mean intensity of phenomenon',
                    'Max intensity of phenomenon','Min intensity of phenomenon','Extent of phenomenon','Axis ratio of phenomenon',
                    'Cross area', 'Cross major axis', 'Cross minor axis']

features = pd.DataFrame(columns = column_features)
path_save_segment = 'C:\\Mitotic Event Detection\\Results\\Segmentation\\'
# for fmp in range(len(full_mito_path)):
fmp = 50
test_path = full_mito_path[fmp]
start_frame = test_path[0]
stop_frame = test_path[-1]
path = test_path[1:-1]
path_ind = 0

distance_factors = []
phe_area_ecc = []
obs_eu = []
max_inten_path = []
mu_inten_path = []
for f in range(start_frame,stop_frame+1):
    props = pd.read_csv(path_area_props + en_props[f])
    lab = cv2.imread(path_lab+en_labs[f],-1)
    im = cv2.imread(path_data + en_data[f],-1)
    c0_area = np.asarray(props['Centroid-0'])[path[path_ind]-1]
    c1_area = np.asarray(props['Centroid-1'])[path[path_ind]-1]
    
    marker = np.zeros(lab.shape,dtype=bool)
    marker[lab == path[path_ind]] = True
    
    invert = util.invert(im)
    coordinates = peak_local_max(invert, min_distance=5, labels = marker)
    
    c0_min = coordinates[:,0]
    c1_min = coordinates[:,1]
    
    pwx = np.power(c0_area - c0_min,2)
    pwy = np.power(c1_area - c1_min,2)
    eu = np.sqrt(pwx+pwy)
    
    mn_eu1 = np.min(eu)
    idx_mn1 = np.argmin(eu)
    eu[idx_mn1] = np.Inf
    mn_eu2 = np.min(eu)
    idx_mn2 = np.argmin(eu)
    obs_eu.append([mn_eu1,mn_eu2])
    
    dis_fac = mn_eu1/mn_eu2
    distance_factors.append(dis_fac)
    
    off = 10
    min_x = props['bbox-1'][path[path_ind]-1] - off
    max_x = props['bbox-3'][path[path_ind]-1] + off
    min_y = props['bbox-0'][path[path_ind]-1] - off
    max_y = props['bbox-2'][path[path_ind]-1] + off
    
    if props['bbox-1'][path[path_ind]-1] - off < 0:
        min_x = props['bbox-1'][path[path_ind]-1]
    if props['bbox-3'][path[path_ind]-1] + off > im.shape[1]:
        max_x = props['bbox-3'][path[path_ind]-1]
    if props['bbox-0'][path[path_ind]-1] - off < 0:
        min_y = props['bbox-0'][path[path_ind]-1]
    if props['bbox-2'][path[path_ind]-1] + off > im.shape[0]:
        max_y = props['bbox-2'][path[path_ind]-1]
    
    cropped_im = im[min_y:max_y,min_x:max_x]
    opt_th_for_phe_im = threshold_li(cropped_im)
    # area_phe_im = cropped_im > opt_th_for_phe_im
    # area_phe_im = cropped_im > 2800
    
    p2_im, p98_im = np.percentile(cropped_im, (2, 95))
    cropped__cs_im = exposure.rescale_intensity(cropped_im, in_range=(p2_im, p98_im))
    
    opt_th_for_phe = threshold_isodata(cropped__cs_im)
    area_phe_im = cropped__cs_im > opt_th_for_phe
    
    lab_cim = measure.label(area_phe_im)
    props_cim = measure.regionprops_table(lab_cim,cropped_im,properties = 
                                                          ['label','area','intensity_mean',
                                                          'axis_major_length','axis_minor_length',
                                                          'eccentricity','intensity_min',
                                                          'intensity_max','coords','centroid'])
    cim_phe_c0 = props_cim['centroid-0']
    cim_phe_c1 = props_cim['centroid-1']
    
    cim_eu = np.sqrt(np.power(c0_area - cim_phe_c0,2) + np.power(c1_area - cim_phe_c1,2))
    phe_area_ecc.append(1 - props_cim['eccentricity'][np.argmin(cim_eu)])
    mu_inten_path.append(props_cim['intensity_mean'][np.argmin(cim_eu)])
    max_inten_path.append(np.max(cropped_im))
    path_ind = path_ind+1

ecc_factors = np.asarray(phe_area_ecc)
dist_factor = np.asarray(distance_factors)
arr_max_inten = np.asarray(max_inten_path)
arr_mean_white_area = np.asarray(mu_inten_path)
intensity_factor = arr_mean_white_area/np.max(arr_max_inten)
arr_combined_factors = np.transpose(np.asarray([distance_factors,intensity_factor,ecc_factors]))

mitosis_factor = 0.1*dist_factor + 0.8*intensity_factor + 0.1*ecc_factors    
mitosis_frame = start_frame + np.argmax(mitosis_factor)
mito_labels = path[np.argmax(mitosis_factor)]
mito_props = pd.read_csv(path_area_props + en_props[mitosis_frame])
mito_labs = cv2.imread(path_lab+en_labs[mitosis_frame],-1)
mito_im = cv2.imread(path_data + en_data[mitosis_frame],-1)

mito_c0_area = np.asarray(mito_props['Centroid-0'])[mito_labels-1]
mito_c1_area = np.asarray(mito_props['Centroid-1'])[mito_labels-1]

 
off = 10
min_x = mito_props['bbox-1'][mito_labels-1] - off
max_x = mito_props['bbox-3'][mito_labels-1] + off
min_y = mito_props['bbox-0'][mito_labels-1] - off
max_y = mito_props['bbox-2'][mito_labels-1] + off

if mito_props['bbox-1'][mito_labels-1] - off < 0:
    min_x = mito_props['bbox-1'][mito_labels-1]
if mito_props['bbox-3'][mito_labels-1] + off > mito_im.shape[1]:
    max_x = mito_props['bbox-3'][mito_labels-1]
if mito_props['bbox-0'][mito_labels-1] - off < 0:
    min_y = mito_props['bbox-0'][mito_labels-1]
if mito_props['bbox-2'][mito_labels-1] + off > mito_im.shape[0]:
    max_y = mito_props['bbox-2'][mito_labels-1]

cropped = mito_im[min_y:max_y,min_x:max_x]

mito_invert = util.invert(mito_im)
mito_marker = np.zeros(mito_labs.shape,dtype=bool)
mito_marker[mito_labs == mito_labels] = True

mito_coordinates = peak_local_max(mito_invert, min_distance=3, labels = mito_marker)

maped_mito_coordinates = np.transpose(np.asarray([mito_coordinates[:,0] - min_y,mito_coordinates[:,1] - min_x]
                                                  ,dtype=np.int64))
mapped_mito_area_c0 = mito_c0_area - min_y
mapped_mito_area_c1 = mito_c1_area - min_x

fig, ax = plt.subplots(figsize=(20,50))
ax.imshow(cropped, cmap = 'gray')
ax.plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax.plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)

mito_eu = np.sqrt(np.power(maped_mito_coordinates[:, 0] - mapped_mito_area_c0,2) + 
                  np.power(maped_mito_coordinates[:, 1] - mapped_mito_area_c1,2))

mito_angle = np.arctan(np.divide(maped_mito_coordinates[:, 1] - mapped_mito_area_c1,
                                  maped_mito_coordinates[:, 0] - mapped_mito_area_c0))


cropped_eq = exposure.equalize_hist(cropped)
cropped_adapeq = exposure.equalize_adapthist(cropped_eq, clip_limit=0.03)

p2, p98 = np.percentile(cropped, (2, 95))
cropped__cs = exposure.rescale_intensity(cropped, in_range=(p2, p98))

fig2, ax2 = plt.subplots(2,2)
fig2.set_figheight(15)
fig2.set_figwidth(20)

ax2[0,0].imshow(cropped)
ax2[0,0].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax2[0,0].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax2[0,0].set_title('Original Cropped Image',fontsize = 30)

ax2[0,1].imshow(cropped_eq)
ax2[0,1].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax2[0,1].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax2[0,1].set_title('Histogram Equalization',fontsize = 30)

ax2[1,0].imshow(cropped_adapeq)
ax2[1,0].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax2[1,0].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax2[1,0].set_title('Adaptive Histogram Equalization',fontsize = 30)

ax2[1,1].imshow(cropped__cs)
ax2[1,1].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax2[1,1].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax2[1,1].set_title('Contrast Streching',fontsize = 30)

block_size = 7
local_thresh = threshold_local(cropped, block_size, offset=10)
binary_local = cropped > local_thresh
inv_bin_local = util.invert(binary_local)

fig3, ax3 = plt.subplots(1,3)
fig3.set_figheight(15)
fig3.set_figwidth(30)

ax3[0].imshow(cropped)
ax3[0].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax3[0].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax3[0].set_title('Original Cropped Image',fontsize = 30)

ax3[1].imshow(cropped__cs)
ax3[1].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax3[1].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax3[1].set_title('Contrast Stretching',fontsize = 30)

ax3[2].imshow(inv_bin_local)
ax3[2].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax3[2].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax3[2].set_title('Invert Binary Image',fontsize = 30)

rev_sobj = smh.remove_small_objects(inv_bin_local,min_size = 10)
rev_shl = smh.remove_small_holes(rev_sobj)
        
labs_a = measure.label(rev_shl)
mito_coor_area_props = measure.regionprops_table(labs_a,properties = ['coords','label'])
mito_coor_area = mito_coor_area_props['coords']                           
lab_for_area_filt = mito_coor_area_props['label']

mask_area = np.zeros(rev_shl.shape,dtype=bool)
for a in range(len(mito_coor_area)):
    cons_area = mito_coor_area[a]
    for c in range(len(maped_mito_coordinates)):
        if (maped_mito_coordinates[c][0] in cons_area[:,0]
            and maped_mito_coordinates[c][1] in cons_area[:,1]):
            mask_area[labs_a == lab_for_area_filt[a]] = True

    
fig4, ax4 = plt.subplots(2,2)
fig4.set_figheight(20)
fig4.set_figwidth(20)

ax4[0,0].imshow(inv_bin_local)
ax4[0,0].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax4[0,0].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax4[0,0].set_title('Invert Binary Image',fontsize = 30)

ax4[0,1].imshow(rev_sobj)
ax4[0,1].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax4[0,1].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax4[0,1].set_title('Remove small objects',fontsize = 30)

ax4[1,0].imshow(rev_shl)
ax4[1,0].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax4[1,0].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax4[1,0].set_title('Remove small holes',fontsize = 30)

ax4[1,1].imshow(mask_area)
ax4[1,1].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax4[1,1].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax4[1,1].set_title('Kept area',fontsize = 30)


distance = ndi.distance_transform_edt(mask_area)
coords_wtf = peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask_area
                            ,min_distance = 5)
mask = np.zeros(rev_shl.shape,dtype=bool)
mask[tuple(coords_wtf.T)] = True    
markers, _ = ndi.label(mask)
watershed_tf = watershed(-distance, markers, mask=mask_area)

# fig6, ax6 = plt.subplots()
# ax6.imshow(cropped)
# ax6.plot(coords_wtf[:, 1], coords_wtf[:, 0], 'r.',markersize = 20)


pwx = np.power(c0_area - c0_min,2)
pwy = np.power(c1_area - c1_min,2)
eu = np.sqrt(pwx+pwy)

mn_eu1 = np.min(mito_eu)
idx_mn1 = np.argmin(mito_eu)
mito_eu[idx_mn1] = np.Inf
mn_eu2 = np.min(mito_eu)
idx_mn2 = np.argmin(mito_eu)
mito_eu[idx_mn2] = np.Inf
mn_eu3 = np.min(mito_eu)
idx_mn3 = np.argmin(mito_eu)
mito_eu[idx_mn3] = np.Inf

closest_point = np.asarray([maped_mito_coordinates[idx_mn1,:],
                maped_mito_coordinates[idx_mn2,:],maped_mito_coordinates[idx_mn3,:]])


fig5, ax5 = plt.subplots(1,2)
fig5.set_figheight(20)
fig5.set_figwidth(20)

ax5[0].imshow(cropped)
ax5[0].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax5[0].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax5[0].plot(closest_point[:,1], closest_point[:,0], 'y.',markersize = 20)
ax5[0].set_title('Original',fontsize = 30)
ax5[0].get_xaxis().set_visible(False)
ax5[0].get_yaxis().set_visible(False)


ax5[1].imshow(watershed_tf)
ax5[1].plot(maped_mito_coordinates[:, 1], maped_mito_coordinates[:, 0], 'r.',markersize = 20)
ax5[1].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)
ax5[1].plot(closest_point[:,1], closest_point[:,0], 'y.',markersize = 20)
ax5[1].set_title('Watershed Transform',fontsize = 30)
ax5[1].get_xaxis().set_visible(False)
ax5[1].get_yaxis().set_visible(False)

fig6, ax6 = try_all_threshold(cropped__cs, figsize=(10, 8), verbose=False)
plt.show()

opt_th_for_phe = threshold_isodata(cropped__cs)
# area_phe = cropped__cs > opt_th_for_phe
# area_phe_ft = 
# ero_area_phe = smh.binary_erosion(area_phe,smh.disk(3))
area_phe = cropped__cs > 60000
rev_phe = smh.remove_small_objects(area_phe,min_size = 20)



phe_lab = measure.label(rev_phe)
phe_props = measure.regionprops_table(phe_lab,cropped,properties = 
                                                  ['label','area','intensity_mean',
                                                  'axis_major_length','axis_minor_length',
                                                  'eccentricity','intensity_min',
                                                  'intensity_max','coords','centroid',
                                                  'extent','bbox'])
pd_phe_props = pd.DataFrame(data = phe_props)

fig7, ax7 = plt.subplots(1,3)
ax7[0].imshow(cropped__cs,cmap = 'gray')
ax7[1].imshow(rev_phe,cmap = 'gray')
ax7[2].imshow(phe_lab)
ax7[2].plot(mapped_mito_area_c1, mapped_mito_area_c0, 'b.',markersize = 20)

phe_cen_area_c0 = np.asarray(pd_phe_props['centroid-0'])
phe_cen_area_c1 = np.asarray(pd_phe_props['centroid-1'])

phe_eu = np.sqrt(np.power(phe_cen_area_c0 - mapped_mito_area_c0,2) + 
          np.power(phe_cen_area_c1 - mapped_mito_area_c1,2))

chosen_phe = pd_phe_props.loc[pd_phe_props.index[pd_phe_props['centroid-0'] == 
                                              phe_cen_area_c0[np.argmin(phe_eu)]]]

# if fmp < 10:
#     pre_vid = '00'
# if fmp >=10 and fmp < 100:
#     pre_vid = '0'
# if fmp > 100:
#     pre_vid = ''

# plt.savefig(path_save_segment + pre_vid + str(fmp) + '.png')
# plt.close()

cells_props = measure.regionprops_table(watershed_tf,cropped,properties = 
                                                  ['label','area','intensity_mean',
                                                  'axis_major_length','axis_minor_length',
                                                  'eccentricity','intensity_min',
                                                  'intensity_max','coords','centroid'])


pd_cell_props = pd.DataFrame(cells_props)
Area_amount = 0

l_cp = list(closest_point)
l_mmc = list(maped_mito_coordinates)

kept_cell = []
kept_cc = []
lab_wtf = cells_props['label']
cell_coord = cells_props['coords']
for a in range(len(cell_coord)):
    cons_area = list(cell_coord[a])
    for c in range(len(l_mmc)):
        temp_loc_c_in_ca = l_mmc[c] == cons_area
        if ((temp_loc_c_in_ca[:,0] & temp_loc_c_in_ca[:,1]).any() 
            and not (lab_wtf[a] in kept_cell)):
            kept_cell.append(lab_wtf[a])
            
    for cc in range(len(l_cp)):
        temp_loc_cc_in_ca = l_cp[cc] == cons_area
        if ((temp_loc_cc_in_ca[:,0] & temp_loc_cc_in_ca[:,1]).any()
            and not (lab_wtf[a] in kept_cc)):
            kept_cc.append(lab_wtf[a])
            
arr_kept_cc = np.asarray(kept_cc)
arr_kept_cell = np.asarray(kept_cell)

kep_cc_ind = []
for l in range(len(lab_wtf)):
    if lab_wtf[l] in arr_kept_cc:
        kep_cc_ind.append(l)
        
kept_cell_ind = []
for l in range(len(lab_wtf)):
    if lab_wtf[l] in arr_kept_cell:
        kept_cell_ind.append(l)

kept_cell_props = pd_cell_props.loc[kept_cell_ind]
# kept_cell_props = kept_cell_props.drop(kep_cc_ind)

md_cell_props = []
if len(kep_cc_ind) < 3 and len(kept_cell_props) != 0:
    if len(kept_cell_props) > 3:
        c0_cell = kept_cell_props['centroid-0'].to_numpy()
        c1_cell = kept_cell_props['centroid-1'].to_numpy()
        
        pwx = np.power(mapped_mito_area_c0 - c0_cell,2)
        pwy = np.power(mapped_mito_area_c1 - c1_cell,2)
        
        cell_eu = np.sqrt(pwx+pwy)
        idx_for_arr = np.argmin(cell_eu)
        mn_idx_1 = kept_cell_props.index[np.argmin(cell_eu)]
        mn_idx_1 = int(mn_idx_1)
        
        if kept_cell_props['label'][mn_idx_1] in kep_cc_ind:
            cell_eu[idx_for_arr] = np.Inf
            mn_idx_2 = kept_cell_props.index[np.argmin(cell_eu)]
            kep_cc_ind.append(pd_cell_props.index[pd_cell_props['label'] ==
                                                  kept_cell_props['label'][mn_idx_2]]
                              .to_numpy(dtype = int)[0])
        else:
            kep_cc_ind.append(pd_cell_props.index[pd_cell_props['label'] ==
                                                  kept_cell_props['label'][mn_idx_1]]
                              .to_numpy(dtype = int)[0])
        
        
        md_cell_props = pd_cell_props.loc[kep_cc_ind]
        md_cell_props = md_cell_props.drop(md_cell_props.index[md_cell_props['eccentricity'] == 
                                            max(md_cell_props['eccentricity'])])
    elif len(kept_cell_props) == 3:
        md_cell_props = kept_cell_props
        md_cell_props = md_cell_props.drop(md_cell_props.index[md_cell_props['eccentricity'] == 
                                            max(md_cell_props['eccentricity'])])
        
elif len(kep_cc_ind) < 3 and len(kept_cell_props) == 0:
    features_for_one_cell = list(np.zeros(11))
    temp_ft_one_cell = pd.DataFrame(data = [features_for_one_cell],
                                    columns = list(pd_cell_props.columns))
    md_cell_props = pd_cell_props.loc[kep_cc_ind]
    md_cell_props = pd.concat([md_cell_props,temp_ft_one_cell], ignore_index=True)
    
elif len(kep_cc_ind) == 3:        
    md_cell_props = pd_cell_props.loc[kep_cc_ind]
    md_cell_props = md_cell_props.drop(md_cell_props.index[md_cell_props['eccentricity'] == 
                                        max(md_cell_props['eccentricity'])])


if len(md_cell_props) == 0:
    kep_cc_ind = []
    c0_cell = kept_cell_props['centroid-0'].to_numpy()
    c1_cell = kept_cell_props['centroid-1'].to_numpy()
    
    pwx = np.power(mapped_mito_area_c0 - c0_cell,2)
    pwy = np.power(mapped_mito_area_c1 - c1_cell,2)
    
    cell_eu = np.sqrt(pwx+pwy)
    
    mn_eu1 = np.min(cell_eu)
    idx_mn1 = kept_cell_props.index[np.argmin(cell_eu)]
    cell_eu[np.argmin(cell_eu)] = np.Inf
    mn_eu2 = np.min(cell_eu)
    idx_mn2 = kept_cell_props.index[np.argmin(cell_eu)]
    cell_eu[np.argmin(cell_eu)] = np.Inf
    mn_eu3 = np.min(cell_eu)
    idx_mn3 = kept_cell_props.index[np.argmin(cell_eu)]
    cell_eu[np.argmin(cell_eu)] = np.Inf
    
    kep_cc_ind.append(pd_cell_props.index[pd_cell_props['label'] ==
                                          kept_cell_props['label'][idx_mn1]]
                      .to_numpy(dtype = int)[0])
    
    kep_cc_ind.append(pd_cell_props.index[pd_cell_props['label'] ==
                                          kept_cell_props['label'][idx_mn2]]
                      .to_numpy(dtype = int)[0])
    
    kep_cc_ind.append(pd_cell_props.index[pd_cell_props['label'] ==
                                          kept_cell_props['label'][idx_mn3]]
                      .to_numpy(dtype = int)[0])
    
    md_cell_props = pd_cell_props.loc[kep_cc_ind]
    
    
    
    
    
    
md_area = md_cell_props['area']
mc = np.max(md_area)
dc = np.min(md_area)

m_props = md_cell_props.loc[md_cell_props['area'] == mc]
d_props = md_cell_props.loc[md_cell_props['area'] == dc]

# Features from mother and daughter cell

area_m = m_props['area'].to_numpy()[0]
maj_m = m_props['axis_major_length'].to_numpy()[0]
mi_m = m_props['axis_minor_length'].to_numpy()[0]
cir_m = m_props['eccentricity'].to_numpy()[0]
mean_int_m = m_props['intensity_mean'].to_numpy()[0]/np.mean(cropped)
max_int_m = m_props['intensity_max'].to_numpy()[0]/np.mean(cropped)
min_int_m = m_props['intensity_min'].to_numpy()[0]/np.mean(cropped)
axis_ratio_m = mi_m/maj_m;

area_d = d_props['area'].to_numpy()[0]
maj_d = d_props['axis_major_length'].to_numpy()[0]
mi_d = d_props['axis_minor_length'].to_numpy()[0]
cir_d = d_props['eccentricity'].to_numpy()[0]
mean_int_d = d_props['intensity_mean'].to_numpy()[0]/np.mean(cropped)
max_int_d = d_props['intensity_max'].to_numpy()[0]/np.mean(cropped)
min_int_d = d_props['intensity_min'].to_numpy()[0]/np.mean(cropped)
axis_ratio_d = mi_d/maj_d;

area_ratio = area_d/area_m
maj_ratio = maj_d/maj_m
minor_ratio = mi_d/mi_m
cir_ratio = cir_d/cir_m
mint_ratio = mean_int_d/mean_int_m
max_ratio = max_int_d/max_int_m
min_ratio = min_int_d/min_int_m
axis_ratio_ratio = axis_ratio_d/axis_ratio_m

    


area_diff = area_m - area_d
maj_diff = maj_m - maj_d
minor_diff = mi_m - mi_d
cir_diff = cir_m - cir_d
mint_diff = mean_int_m - mean_int_d
max_diff = max_int_m - max_int_d
min_diff = min_int_m - min_int_d
axis_ratio_diff = axis_ratio_d - axis_ratio_m

# Features from area of phenomenon

area_phe = chosen_phe['area'].to_numpy()[0]
maj_phe = chosen_phe['axis_major_length'].to_numpy()[0]
mi_phe = chosen_phe['axis_minor_length'].to_numpy()[0]
cir_phe = chosen_phe['eccentricity'].to_numpy()[0]
mean_int_phe = chosen_phe['intensity_mean'].to_numpy()[0]/np.mean(cropped)
max_int_phe = chosen_phe['intensity_max'].to_numpy()[0]/np.mean(cropped)
min_int_phe = chosen_phe['intensity_min'].to_numpy()[0]/np.mean(cropped)
ext_phe = chosen_phe['extent'].to_numpy()[0]/np.mean(cropped)
axis_ratio_phe = mi_phe/maj_phe;

# Cross features

cross_area = (area_m + area_d)/area_phe
cross_maj = (maj_m + maj_d)/maj_phe
cross_mi = (mi_m + mi_d)/mi_phe

# Coverage phenomenon area to detected cell
m_coord = m_props['coords'].to_numpy()[0]
d_coord = d_props['coords'].to_numpy()[0]

m_cov0 = np.logical_and(m_coord[:,0] >= chosen_phe['bbox-0'].to_numpy()[0],
                        m_coord[:,0] <= chosen_phe['bbox-2'].to_numpy()[0])

m_cov1 = np.logical_and(m_coord[:,1] >= chosen_phe['bbox-1'].to_numpy()[0],
                        m_coord[:,1] <= chosen_phe['bbox-3'].to_numpy()[0])

d_cov0 = np.logical_and(d_coord[:,0] >= chosen_phe['bbox-0'].to_numpy()[0],
                        d_coord[:,0] <= chosen_phe['bbox-2'].to_numpy()[0])

d_cov1 = np.logical_and(d_coord[:,1] >= chosen_phe['bbox-1'].to_numpy()[0],
                        d_coord[:,1] <= chosen_phe['bbox-3'].to_numpy()[0])

percentage_cov_m = np.sum(np.logical_and(m_cov0,m_cov1))/len(m_cov0)
percentage_cov_d = np.sum(np.logical_and(d_cov0,d_cov1))/len(d_cov0)


features_list = [area_m,maj_m,mi_m,cir_m,mean_int_m,max_int_m,min_int_m,axis_ratio_m,
            area_d,maj_d,mi_d,cir_d,mean_int_d,max_int_d,min_int_d,axis_ratio_d,
            area_ratio,maj_ratio,minor_ratio,cir_ratio,mint_ratio,max_ratio,min_ratio,axis_ratio_ratio,
            area_diff,maj_diff,minor_diff,cir_diff,mint_diff,max_diff,min_diff,axis_ratio_diff,
            area_phe,maj_phe,mi_phe,cir_phe,mean_int_phe,max_int_phe,min_int_phe,ext_phe,axis_ratio_phe,
            cross_area,cross_maj,cross_mi]




temp_features_frame = pd.DataFrame(data = [features_list],columns = column_features)
features = pd.concat([features,temp_features_frame], ignore_index=True)