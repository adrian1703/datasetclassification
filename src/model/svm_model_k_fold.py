# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:10:32 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sklearn import svm
from sklearn import preprocessing
from sklearn.utils import shuffle
from model.k_fold_helper import get_split, iterate_over_corpus
import pandas as pd
import numpy as np


#%%
def get_train_test_split(ddc_class, df, k_fold_iter, splits, random_state=25):
    train_df, test_df = get_split(k_fold_iter, df, splits=splits, drop_duplicates_df=True, shuffle_df=True, random_state=random_state)
    return (train_df, test_df)
    
def merge_duplicates_test_data(test_data):
    test_topics = test_data.groupby("page_uri")["topic_id"].apply(list).reset_index(name="actual")
    test_other = test_data.groupby("page_uri")[["abstract_word_count", "abstract_character_count", "tokens"]].first()
    result = pd.merge(test_topics, test_other, on="page_uri")
    result.reset_index(drop=True)
    return result

def get_global_stats(test_data_y, predicted_y, top_k=10, thresholds=[i/10 for i in [*range(0, 10, 1)]]):
    results = []
    for threshold in thresholds:
        data = {"type":"global",
                "threshold":threshold, 
                "total": len(test_data_y), 
                "unclass":0}
        for k in [*range(1,top_k+1,1)]:
            data.setdefault(k, 0)
        for entry_test, entry_predicted in list(zip(test_data_y, predicted_y)):
            possible_labels = [label for weight, label in entry_predicted if weight > threshold]
            if len(possible_labels) == 0:
                data["unclass"] += 1
                continue
            for k in [*range(1,top_k+1,1)]:
                intersection = set(entry_test) & set(possible_labels[:k])
                if intersection:
                    data[k]+=1
        for k in [*range(1,top_k+1,1)]:
            d = data["total"] - data["unclass"]
            if d != 0:
                data[k] /= data["total"] - data["unclass"]
            
        data["unclass"] /= data["total"]
        results.append(data)
                
            
    return pd.DataFrame(results)

def get_class_stats(test_data_y, predicted_y,ddc_classes_to_include, top_k=10, thresholds=[i/10 for i in [*range(0, 10, 1)]]):
    results = []
    x = []
    for i in test_data_y:
        x += i
    for ddc in sorted(list(set(ddc_classes_to_include) & set(x))):
        predicted_y_ddc = [i[1] for i in list(zip(test_data_y, predicted_y)) if ddc in i[0]]
        for threshold in thresholds:
            data = {"type":ddc,
                    "threshold":threshold, 
                    "total": len(predicted_y_ddc), 
                    "unclass":0}
            for k in [*range(1,top_k+1,1)]:
                data.setdefault(k, 0)
            for entry_predicted in predicted_y_ddc:
                possible_labels = [label for weight, label in entry_predicted if weight > threshold]
                if len(possible_labels) == 0:
                    data["unclass"] += 1
                    continue
                for k in [*range(1,top_k+1,1)]:
                    if ddc in possible_labels[:k]:
                        data[k]+=1
            for k in [*range(1,top_k+1,1)]:
                d = data["total"] - data["unclass"]
                if d != 0:
                    data[k] /= data["total"] - data["unclass"]
            if data["total"] != 0:
                data["unclass"] /= data["total"]
            results.append(data)
                
    results =   pd.DataFrame(results)    
    return results.query("total!=0")

def k_fold_svm(stm_models, corpus_dir, ddc_classes_to_include = [*range(0,1000,10)], svm_type= "linear", 
               svm_C=0.1, svm_gamma="scale", svm_degree=3, svm_coef0=0,
               random_state=25, train_chunk=10000, early_break=False):
    
    svms = []
    global_res = []
    splits = len(stm_models) if not early_break else 1
    for k_fold_it in [*range(splits)]:
        print(k_fold_it)
        current_model = stm_models[k_fold_it]
        
        print("Preparing trainings data")
        train_test_split = lambda ddc_class, df: get_train_test_split(ddc_class, df, k_fold_it, splits=5, random_state=random_state)
        train_test_data = iterate_over_corpus(corpus_dir, train_test_split, ddc_classes_to_include)
        
        train_data = pd.concat([i[0] for i in train_test_data])
        train_data = shuffle(train_data, random_state=25)[:min(train_chunk, len(train_data.index))]
        train_data_x = [current_model.get_class_scores_tokens(row.tokens) for row in train_data.itertuples()]
        train_data_y = np.array([i//10 * 10  for i in train_data[["topic_id"]].values.ravel()]).reshape(-1,1)
        
        scaler_x = preprocessing.StandardScaler().fit(train_data_x,train_data_y)
        train_data_x_scaled = scaler_x.transform(train_data_x)

        print("Training svm")
        svc = svm.SVC(kernel=svm_type, C=svm_C, gamma=svm_gamma, degree=svm_degree, coef0=svm_coef0, probability=True, decision_function_shape="ovr")
        svc.fit(train_data_x_scaled, train_data_y.ravel())
        
        print("Preparing test data")
        test_data = pd.concat([i[1] for i in train_test_data])
        test_data_merged = merge_duplicates_test_data(test_data)
        test_data_x = [current_model.get_class_scores_tokens(row.tokens) for row in test_data_merged.itertuples()]
        test_data_x_scaled = scaler_x.transform(test_data_x)
        test_data_y = [[i//10 * 10 for i in row.actual] for row in test_data_merged.itertuples()]
        predicted_y = svc.predict_proba(test_data_x_scaled)
        predicted_y = [sorted(list(zip(weight, svc.classes_)), reverse=True)[:10]for weight in predicted_y]
        
        print("Evaluating test data")
        global_res.append(pd.concat([get_global_stats(test_data_y, predicted_y), get_class_stats(test_data_y, predicted_y, ddc_classes_to_include)]))
        svms.append((svc, scaler_x))



    return global_res, svms
#%%
# f = lambda df, stm_model: get_x_vec(df, stm_model, token_count=False, abstract_word_count=False, abstract_character_count=False)
# a,b = k_fold_svm(models, src_dir, f, train_chunk=100000, early_break=False)
#%%
# def k_fold2_svm(stm_models, corpus_dir, df_to_vec_x_func, svm_type= "linear", 
#                svm_C=0.1, svm_gamma="scale", svm_probability=False,
#                random_state=25, train_chunk=10000, early_break=False, ddc_to_exclude=[]):
    
#     svms = []
#     global_res = []
#     splits = len(stm_models) if not early_break else 1
#     for k_fold_it in [*range(splits)]:
#         print(k_fold_it)
#         current_model = stm_models[k_fold_it]
        
#         train_test_split = lambda ddc_class, df: get_train_test_split(ddc_class, df, k_fold_it, random_state=random_state)
#         train_test_data = iterate_over_corpus(src_dir, train_test_split, ddc_classes_to_exlude)
        
#         train_data = pd.concat([i[0] for i in train_test_data])
#         test_data = pd.concat([i[1] for i in train_test_data])
#         test_data = remove_duplicates_test_data(test_data)
        
#         train_data["token_count"] = get_token_count_col(train_data)
#         test_data ["token_count"] = get_token_count_col(test_data)
        
#         train_data = shuffle(train_data, random_state=25)[:min(train_chunk, len(train_data.index))]
#         train_data_x = df_to_vec_x_func(train_data, current_model)
#         train_data_y = np.array([i//10 * 10 if i not in ddc_to_exclude else -1 for i in train_data[["topic_id"]].values.ravel() ]).reshape(-1,1)
        
#         scaler_x = preprocessing.StandardScaler().fit(train_data_x,train_data_y)
#         x_scaled = scaler_x.transform(train_data_x)
        
#         train_data_x_scaled = scaler_x.transform(train_data_x)
        
#         test_data_x_scaled = df_to_vec_x_func(test_data, current_model)
#         test_data_y = [[i//10 * 10 if i not in ddc_to_exclude else -1 for i in row.actual] for row in test_data.itertuples()]
        
#         svc = svm.SVC(kernel=svm_type, C=svm_C, gamma=svm_gamma, probability=svm_probability, decision_function_shape="ovr")
#         svc.fit(train_data_x_scaled, train_data_y.ravel())
        
#         predicted_y = svc.predict(scaler_x.transform(test_data_x_scaled))
        
#         pos = []
#         for i in list(zip(predicted_y, test_data_y)):
#             if i[0] in i[1]:
#                 pos.append(i)
#         global_res.append(len(pos)/len(test_data_y))
#         svms.append(svc)
#         if early_break:
#             break
#     return global_res, svms

# path = "D:/Informatikstudium/bachelor/stp/src/model_eval_scripts/testcase_ranking_and_threshold_result.csv"
# a = pd.read_csv(path, sep=";")
# b = a.query("ranking_func=='ranking_c' and threshold==0 and ( `1` > 0.4  or `2` > 0.6)")
# ddc_classes_to_include = set(b["type"].apply(int))
# all_ddc_classes = set([*range(0,1000,10)])
# ddc_classes_to_exlude = all_ddc_classes - ddc_classes_to_include
# f = lambda df, stm_model: get_x_vec(df, stm_model, token_count=False, abstract_word_count=False, abstract_character_count=False)
# a2,b2 = k_fold2_svm(models, src_dir, f, train_chunk=100000, early_break=True, ddc_to_exclude=ddc_classes_to_exlude)
# f = lambda df, stm_model: get_x_vec(df, stm_model, token_count=False, abstract_word_count=False, abstract_character_count=False)
# svm, res, test_x, test_y = k_fold2_svm(models, src_dir, f, train_chunk=200000, early_break=True, svm_probability=True)
#%% 0.69 - 10 000
#%% 0.72