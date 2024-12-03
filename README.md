# Version 1.0.0
## Introduction
Welcome to the mitotic event detection project for the lens-free imaging image dataset. This project contains 2 main code files for the features extraction stored in the "Core" folder. Code for the extracted features inspection and classification test is also included in that folder. The dataset can be downloaded from the link https://osf.io/ysaq2/ by the step mentioned in the download instruction https://osf.io/vuj9w 

## CORE function
The code for POC is core.py and the code for features extraction is core_rerun.py which contains subprocesses:
- image preprocessing for cell detection in the intensity image
- Events pathway generation by linking the cell in each image frame
- Path similarity calculation to eliminate the path that has similar positions more than 40%
- Features extraction from the area of interest
- Features extraction from mother cell and daughter cell in telophase frame
- Features extraction from cell moving.

## Unit Module for Feature Extraction Process Observation

This document provides an overview of the procedures for running unit modules in this project. Each module generates results corresponding to the processes outlined in the associated publication, including image preprocessing, segmentation, event tracking, and feature extraction.

The results are categorized into two datasets:
1. A threshold value of 2800 is applied for AOI segmentation.
2. A threshold value of 2500 is applied for AOI segmentation, resulting in longer event paths.

Details about the functions in the **Code** folder are as follows:
- **`segmentation.py`**: Executes the AOI segmentation process, extracting region properties and labeled images.
- **`mitosis_path_generation.py`**: Handles event tracking and generates lists of event paths for each sequence.
- **`inner_AOI_segmentation_new.py`**: Observes cell segmentation within AOIs for telophase frames.

## Labels and Ground Truth Videos

The **Label** folder contains aligned labels for the ground truth videos provided in this project. Subfolders **Seq1** to **Seq4** are included in both the **Label** and ground truth video directories. These videos are generated based on event tracking paths from the **`mitosis_path_generation.py`** script. Each sequence contains a `label.csv` file with the following columns:
- `Label`: The unique identifier for each tracked region.
- `Class`: The classification of the tracked region.
- `Reference File`: The associated file for the tracked region.

## Event Path Observation

Event tracking may require significant computational time to complete. Precomputed event paths are stored in the folders:
- **Full mito path**
- **Full mito path 2500th**

You can observe these paths using the **`path_observation.py`** script. The loaded paths are stored as NumPy arrays in the format:  
`[Start Frame | Order of the Cell Region (referencing region properties for each frame) | End Frame]`

## Label Images

The labeled images are stored in the following directories based on the threshold values used during image segmentation:

- **`Label pic`**: Contains labeled images with a 2800 threshold.  
- **`Label pic 2500th`**: Contains labeled images with a 2500 threshold.

These labeled images are utilized for extracting region properties, which serve as input for the event tracking process. The extracted region properties are saved in the respective directories:

- **`Region properties`**: Corresponding to the `Label pic` folder (2800 threshold).  
- **`Region properties 2500th`**: Corresponding to the `Label pic 2500th` folder (2500 threshold).


## How to Download Data from the PCM Dataset

1. Visit the [PCM dataset](https://osf.io/ysaq2/).
2. Select the file you wish to download. For example, files marked within the red square in the provided figure. ![step 2](https://github.com/Plakaeze/Mitotic-event-detection-project-PCM/blob/main/How%20to%20download%20PCM/step_1.png?raw=true)
3. Once selected, the interface will display additional details. Click the pop-up button highlighted in red in the figure. ![step 3](https://github.com/Plakaeze/Mitotic-event-detection-project-PCM/blob/main/How%20to%20download%20PCM/step_2.png?raw=true)
4. To download multiple files or entire folders, use the button within the red square in the panel. ![step 4](https://github.com/Plakaeze/Mitotic-event-detection-project-PCM/blob/main/How%20to%20download%20PCM/step_3.png?raw=true)
