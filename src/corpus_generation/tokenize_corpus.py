# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:59:22 2023

@author: Adrian Kuhn
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import glob
import os.path
import model.tokenize_helper as th
import model.model_construction_helper as mh
#%%
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc"
targed_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized"
src_folders = glob.glob(f"{src_dir}/*")

for src_folder in src_folders:
    suffix = src_folder.split("\\")[-1]
    target_folder = f"{targed_dir}/{suffix}" 
    if not os.path.isfile(target_folder):
        os.mkdir(target_folder)
    f = lambda ddc_class, df : th.tokenize_df(ddc_class, df, target_folder)
    mh.iterate_over_corpus(src_folder, f)
   