# Version 1.0.0
## Introduction
Welcome to the mitotic event detection project for the lens-free imaging image dataset. To request the dataset, please send an email to manorost.panason@kuleuven.be because the dataset is not yet publicly available. This project contains 2 main code files for the features extraction stored in the "Core" folder. Code for the extracted features inspection and classification test is also included in that folder. The dataset can be downloaded from the link https://osf.io/ysaq2/ by the step mentioned in the download instruction https://osf.io/vuj9w 

## CORE function
The code for POC is core.py and the code for features extraction is core_rerun.py which contains subprocesses:
- image preprocessing for cell detection in the intensity image
- Events pathway generation by linking the cell in each image frame
- Path similarity calculation to eliminate the path that has similar positions more than 40%
- Features extraction from the area of interest
- Features extraction from mother cell and daughter cell in telophase frame
- Features extraction from cell moving.

## Unit module running for features extraction process observation
This section describe the unit runing procedure for this project. Each unit can provide the results of each process mentioned in the publication which include image preprocessing and segmentation, event tracking, and features extraction. The reference of the results are divided into 2 sets. First set we apply a value 2800 for threshold of AOI segmentation and another set we apply a value 2500 to the AOI segmentation which cause longer events path. The unit code function in the "Code" folder are explained below.
- segmentation.py is the script for the AOI segmentation process which extract both region properties and label images.
- mitosis_path_generation.py is the script for the event tracking process, this script generate the list of the event path for each sequence.
- inner_AOI_segmentation_new.py is the script to observe the cells segmentation within AOI foe the telophase frame.

## Label and ground truth video
The label folder is created to aligned on the video for ground truth file uploaded in this project. The folder of the ground truth video contains sub folder Seq1 to Seq4 similar to the label folder. These video are generated from the event tracking path that provided by the mitosis_path_generation unit. The label.csv in each sequence contain the label, class, and reference file information

## Event path observation
Since the event tracking take long time to finish, the event path are stored in the folder "Full mito path" and "Full mito path 2500th". The list of path can be loaded and observe using the script path_observation.py. The loaded list contain the numpy array in the format [Start Frame|order of the cell region refer to the region properties of each frame|End Frame]

