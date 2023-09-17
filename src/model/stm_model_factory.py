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
import math
import model.stm_model as stm
from model.k_fold_helper import get_split

############### For single usage test ########################################
# Usage
# corpus_dir = "..."
# class_models, testset = construct_class_models(corpus_dir)
# stm_model = construct_stm_model(corpus_dir, class_models)

def get_stm_model(corpus_dir, ddc_classes=[*range(0,1000,10)]):
    results, test_set = construct_class_models(corpus_dir, ddc_classes)
    stm_model = construct_class_models(corpus_dir, class_models, ddc_classes)
    return stm_model, test_set

def construct_class_models(corpus_dir, ddc_classes): 
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
        class_model = stm.DdcClassModel(ddc_class=ddc_class)
        class_model.calc_model(train)
        results.append(class_model)
    return results, test

def construct_stm_model(corpus_dir, class_models, ddc_classes):
    stm_model = stm.StmModel()
    stm_model.construct_model(class_models)
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


    
    

##############################################################################


############### For K Fold usage #############################################

def construct_class_models_k_fold(ddc_class, df, k_fold_iter, random_state):
    
    train_df, test_df = get_split(k_fold_iter, df, drop_duplicates_df=True, shuffle_df=True, random_state=random_state)
    class_model = stm.DdcClassModel(ddc_class=ddc_class)
    class_model.calc_model(train_df)
    return class_model

def assign_class_scores_stm_model_k_fold(ddc_class, df, k_fold_iter, stm_model, random_state):
    
    train_df, test_df = get_split(k_fold_iter, df, drop_duplicates_df=True, shuffle_df=True, random_state=random_state)
    stm_model.add_class_scores_thresholds(train_df)
    return ddc_class

# calculate ddc_scores for test-pages belonging to given ddc class in a given
# k fold iteration
# used with iterator_over_corpus to calculate scores for all test pages of a corpus
def predict_test_k_fold(ddc_class, df, k_fold_iter, stm_model, random_state, top_k, ranking_functions):
    results = []
    train_df, test_df = get_split(k_fold_iter, df, drop_duplicates_df=True, shuffle_df=True, random_state=random_state)
    
    for row in test_df.itertuples():
        uri = row.page_uri
        tokens = row.tokens
        scores = stm_model.get_class_scores_tokens(tokens)
        percentiles = stm_model.get_percentile_class_scores(scores)
        for rf_name, rf_lambda in ranking_functions.items():
            predicted_classes = stm_model.predict_percentiles(percentiles, top_k=top_k, ranking_function=rf_lambda)
            results.append({
                "page_uri":uri, 
                "actual":ddc_class, 
                "predicted":predicted_classes,
                "ranking_function": rf_name,
                })
    return pd.DataFrame(data=results)

##############################################################################
