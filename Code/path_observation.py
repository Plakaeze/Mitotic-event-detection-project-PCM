# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:55:45 2024

@author: u0158287
"""


import pickle

# Path of the events
sequence = 1
path_events = 'C:\\Mitotic Event Detection\\Full mito path\\Seq' + str(sequence) + '//full_mito_path'

with open(path_events, 'rb') as f:
   path = pickle.load(f)
   
# Each numpy array in the loaded list represent the mitotic path which is kept in the from of
# [Start Frame|order of the cell region refer to the cell properties|End Frame]