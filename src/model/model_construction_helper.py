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

def construct_class_models_k_fold(ddc_class, df, k_fold_iter, random_state=25):
    local_df = df.drop_duplicates().reset_index(drop=True)
    local_df = shuffle(local_df, random_state=random_state)
    train_df, test_df = get_split(k_fold_iter, local_df)
    class_model = model_gen.DdcClassModel(ddc_class=ddc_class)
    class_model.calc_model(train_df)
    return class_model

def assign_class_scores_stm_model_k_fold(ddc_class, df, k_fold_iter, stm_model, random_state=25):
    local_df = df.drop_duplicates().reset_index(drop=True)
    local_df = shuffle(local_df, random_state= random_state)
    train_df, test_df = get_split(k_fold_iter, local_df)
    stm_model.add_class_scores_thresholds(train_df)
    return ddc_class

# For single usage test
# Usage
# corpus_dir = "..."
# class_models, testset = construct_class_models(corpus_dir)
# stm_model = construct_stm_model(corpus_dir, class_models)
def construct_class_models(corpus_dir): 
    ddc_classes = [*range(0,1000,10)]
    results = []
    test = []
    for ddc_class in ddc_classes:
        file_path = glob.glob(f"{corpus_dir}/{ddc_class}.csv")
        if not file_path:
            continue
        df = pd.read_csv(file_path[0], sep=";")
        df = df.drop_duplicates().reset_index(drop=True)
        df = shuffle(df, random_state=(1))
        split = (len(df.index) // 5) * 4
        train = df[:split] 
        test.append(df[split:])
        class_model = model_gen.DdcClassModel(ddc_class=ddc_class)
        class_model.calc_model(train)
        results.append(class_model)
    return results, test

def construct_stm_model(corpus_dir, class_models):
    stm_model = model_gen.StmModel()
    stm_model.construct_model(class_models)
    ddc_classes = [*range(0,1000,10)]
    for ddc_class in ddc_classes:
        file_path = glob.glob(f"{corpus_dir}/{ddc_class}.csv")
        if not file_path:
            continue
        df = pd.read_csv(file_path[0], sep=";")
        df = df.drop_duplicates().reset_index(drop=True)
        df = shuffle(df, random_state=(1))
        split = (len(df.index) // 5) * 4
        train = df[:split] 
        stm_model.add_class_scores_thresholds(train)
    stm_model.calc_threshold_percentiles()
    return stm_model
