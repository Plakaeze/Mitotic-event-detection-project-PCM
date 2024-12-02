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
This section describe the unit runing procedure for this project. Each unit can provide the results of each process mentioned in the publication which include image preprocessing and segmentation, event tracking, and features extraction.
