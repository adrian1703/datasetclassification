# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 00:04:31 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import os.path
import glob
from sklearn.utils import shuffle
import math
import model.model_gen as model_gen
#%%
def iterate_over_corpus(src_folder, func, classes_to_exlude=[]):
    ddc_classes = [*range(0,1000,10)]
    result = []
    for ddc_class in ddc_classes:
        if ddc_class in classes_to_exlude:
            continue
        file_path = glob.glob(f"{src_folder}/{ddc_class}.csv")
        if not file_path:
            continue
        df = pd.read_csv(file_path[0], sep=";")
        
        result.append(func(ddc_class, df))
    return result

def get_split(iteration, df, splits=5):
    split_size = len(df.index) / splits
    indicies = [math.floor(split_size * i) for i in [*range(splits + 1)]]
    chunks = [df[indicies[i] : indicies[i + 1]] for i in [*range(splits)]]
    test = chunks.pop(iteration)
    train = pd.concat(chunks)
    return train, test

