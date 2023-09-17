# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:01:38 2023

@author: Adrian Kuhn
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pandas as pd
import os.path
import ast
from dataclasses import dataclass, field
import numpy as np
import math
import model.tokenize_helper as tokenize_helper
#%%

@dataclass()
class DdcClassModel:
    ddc_class: int = field()
    class_token_dict: dict = field(default_factory=dict)
    class_score_thresholds: dict = field(default_factory=dict)
    
    
    def calc_model(self, df, threshold_steps=1):
        self.__sum_tokens(df)
        self.__trim_rare_tokens()
        self.__normalize_token_occ()
        self.class_token_dict = dict(sorted(self.class_token_dict.items()))
        
        
    def __sum_tokens(self,df):
        for row in df.itertuples():
            page_tokens = ast.literal_eval(row.tokens) if type(row.tokens) == str else row.tokens
            for token, count in page_tokens.items():
                self.class_token_dict.setdefault(token, 0)
                self.class_token_dict[token] += count
                
    def __trim_rare_tokens(self, occurence_threshold=-1):
        if occurence_threshold == -1:
            avg_token_count = sum(self.class_token_dict.values())/len(self.class_token_dict.values())
            occurence_threshold = avg_token_count // 3
        self.class_token_dict = {key : value for key, value in self.class_token_dict.items() if value >= occurence_threshold}
        
    def __normalize_token_occ(self):
        total_tokens = sum(self.class_token_dict.values())
        self.class_token_dict = {key : value/total_tokens for key, value in self.class_token_dict.items()}
        

@dataclass()
class StmModel:
    model: dict = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    ddc_classes: list[int] = field(default_factory=list)
    class_threshholds_pos: dict = field(default_factory=dict)
    class_threshholds_neg: dict = field(default_factory=dict) 
    ranking_function = lambda a: a[0]   
    ranking_threshold = 0
    
    def construct_model(self, ddc_class_model_list):
        self.__assign_tokens(ddc_class_model_list)
        self.__assign_ddc_classes(ddc_class_model_list)
        self.__construct_model_matrix(ddc_class_model_list)
        self.__init_threshold_dicts()
        
    
    def __assign_tokens(self, ddc_class_model_list):
        self.tokens = [i.class_token_dict for i in ddc_class_model_list]
        self.tokens = list(set().union(*self.tokens))
        self.tokens.sort()
    
    def __assign_ddc_classes(self, ddc_class_model_list):
        self.ddc_classes = [i.ddc_class for i in ddc_class_model_list]
        self.ddc_classes.sort()
    
        
    def __construct_model_matrix(self, ddc_class_model_list):
        for token in self.tokens:
            self.model.setdefault(token,  [0 for i in self.ddc_classes])
            for class_model in ddc_class_model_list:
                ddc_class_i = self.get_class_i(class_model.ddc_class)
                if token in class_model.class_token_dict.keys():
                    self.model[token][ddc_class_i] = class_model.class_token_dict[token]
            self.model[token] = np.array(self.model[token])
            
    def __init_threshold_dicts(self):
        for i in self.ddc_classes:
            self.class_threshholds_pos.setdefault(i, [])
            self.class_threshholds_neg.setdefault(i, [])
            
    # Adds class scores from training sets for percentile distribution       
    def add_class_scores_thresholds(self, testset, convert_str_to_token=True):
        for row in testset.itertuples():
            page_tokens = ast.literal_eval(row.tokens) if convert_str_to_token else row.tokens
            if not page_tokens: continue
            topic = (row.topic_id // 10) * 10 
            scores = self.get_class_scores_tokens(page_tokens, convert_str_to_token=False)
            for i in [*range(len(self.ddc_classes))]:
                current_class = self.ddc_classes[i]
                current_score = scores[i]
                if int(topic) == current_class:
                    self.class_threshholds_pos[current_class].append(current_score)
                else:
                    self.class_threshholds_neg[current_class].append(current_score)
    # Calculate percentile thesholds based on the train class scores and
    # their distribution            
    def calc_threshold_percentiles(self):
        self.__calc_threshold_percentiles(self.class_threshholds_pos)
        self.__calc_threshold_percentiles(self.class_threshholds_neg)
        
    def __calc_threshold_percentiles(self, threshold_dict):
        for key, values in threshold_dict.items():
                data = sorted(values)
                step_size = len(values) / 100 
                data_new = {i:0 for i in [*range(1,100,1)]}
                for i in [*range(1,100,1)]:
                    index = math.floor(step_size * i) 
                    data_new[i] = data[index]
                threshold_dict[key] = data_new
    
    # Scores a text document
    def get_class_scores_doc(self, doc):
        page_tokens = tokenize_helper.tokenize_gensim(doc)
        page_tokens = tokenize_helper.count_tokens(page_tokens)
        return self.get_class_scores_tokens(page_tokens, convert_str_to_token=False)
    
    # Scores a token : occurence list
    def get_class_scores_tokens(self, page_tokens, convert_str_to_token=True):
        _tokens = ast.literal_eval(page_tokens) if convert_str_to_token else page_tokens
        weights = [np.zeros(shape=(len(self.ddc_classes)), dtype=float)]
        normalize_factor = max(sum(_tokens.values()), 1)
        for token, count in _tokens.items():
            if token in self.model.keys():
                weights.append(count * self.model[token])
        scores = sum(weights) / normalize_factor
        return scores
    
    # Returns the percentile score for class scores
    def get_percentile_class_scores(self, scores):
        percentile_scores = []
        for i in [*range(len(scores))]:
            score = scores[i]
            current_class = self.ddc_classes[i]
            pos = self.__get_percentile(score, self.class_threshholds_pos[current_class])
            neg = self.__get_percentile(score, self.class_threshholds_neg[current_class])
            percentile_scores.append((pos,neg))
        return percentile_scores
    
    def __get_percentile(self, score, thresholddict):
        percentile = 0
        for current_percentile, value in sorted(thresholddict.items()):
            percentile = current_percentile
            if value >= score:
                break
        return percentile
    
    # Predict top_k classes for a text_document
    # Apply a percentile_function to the scores tuple ((a,b), g) -> (c, g)
    # Sort the sorces by c descending an return the top k scores that are 
    #   above the threshold
    # scores:           list of ((a,b), g)
    # ranking_func:  lambda ((a,b), g) -> (c,g)
    # threshold:        number
    # top_k:            number | how many top entries to fetch
    def predict_doc(self, doc, top_k=10, ranking_function=lambda a,b: a * b / 100, ranking_threshold=0):
        scores = self.get_class_scores_doc(doc)
        return self.__predict(scores, top_k, ranking_function, ranking_threshold)
    
    def predict_tokens(self, page_tokens, convert_str_to_token=True, top_k=10, ranking_function=lambda a,b: a * b / 100, ranking_threshold=0):
        scores = self.get_class_scores_tokens(page_tokens, convert_str_to_token)
        return self.__predict(scores, top_k, ranking_function, ranking_threshold)
    
    # Only useful for k-fold 
    def predict_percentiles(self, percentile_scores, top_k, ranking_function):
        data = list(zip(percentile_scores, self.ddc_classes))
        data = [(ranking_function(i[0][0], i[0][1]), i[1]) for i in data]
        data = sorted(data, reverse=True)
        data = data[:top_k]
        return tuple(data)
    
    def __predict(self, scores, top_k, ranking_function, ranking_threshold):
        percentile_scores = self.get_percentile_class_scores(scores)
        data = list(zip(percentile_scores, self.ddc_classes))
        data = [(ranking_function(i[0][0], i[0][1]), i[1]) for i in data]
        data = sorted(data, reverse=True)
        data = [i for i in data[:top_k] if i[0] > ranking_threshold]
        return tuple(data)

                    
            
    



    
    def get_token_i(self, token, start=0):
        return self.__binary_search(self.tokens, token, start)
    
    def get_class_i(self, ddc_class):
        return self.__binary_search(self.ddc_classes, ddc_class)
    
    def __binary_search(self, arr, x, start=0):
        low = start
        high = len(arr) - 1
        mid = start
        while low <= high:
            mid = (high + low) // 2
            if arr[mid] < x:
                low = mid + 1
            elif arr[mid] > x:
                high = mid - 1
            else:
                return mid
        return -1
    





