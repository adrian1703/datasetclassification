# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:55:47 2023

@author: Adrian Kuhn
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.k_fold_helper import get_split, iterate_over_corpus
import pandas as pd
import csv
import fasttext
from gensim.utils import simple_preprocess
import io
import model.svm_model_k_fold as svm_model_k_fold
#%%
def get_train_test_split(ddc_class, df, k_fold_iter, splits, random_state=25):
    train_df, test_df = get_split(k_fold_iter, df, splits=splits, drop_duplicates_df=True, shuffle_df=True, random_state=random_state)
    return (train_df, test_df)


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def preprocess_data(dataset, file_path, simple_preprocess_=False):
    dataset["topic_id"] = dataset["topic_id"].apply(lambda topic: topic // 10  * 10)
    dataset = dataset.groupby(["page_uri","abstract"])["topic_id"].apply(set).apply(list).reset_index(name="topics")
    dataset["fast_text_label"] = dataset["topics"].apply(lambda topics: " ".join([f"__label__{topic}" for topic in topics]))
    if simple_preprocess_:
        dataset["abstract2"] = dataset["abstract"].apply(lambda abstract: " ".join(simple_preprocess(abstract)))
    if not os.path.exists(file_path):
        with open(file_path, 'w'): pass
    dataset[["fast_text_label", "abstract"]].to_csv(file_path, 
                                              index = False, 
                                              sep = ' ',
                                              header = None, 
                                              quoting = csv.QUOTE_NONE, 
                                              quotechar = "", 
                                              escapechar = " ")

def get_global_stats(test_data, model, top_k=10, thresholds=[i/10 for i in [*range(0, 10, 1)]]):
    results = []
    test_data_local = test_data.copy(deep=True)
    test_data_local["topic_id"] = test_data_local["topic_id"].apply(lambda topic: topic // 10  * 10)
    test_data_local = test_data_local.groupby(["page_uri","abstract"])["topic_id"].apply(set).apply(list).reset_index(name="topics")
    data_points = []
    for test_entry in test_data_local.itertuples():
        abstract = (test_entry.abstract).replace("\n", " ")
        labels = test_entry.topics
        predicted_labels = model.predict(abstract, k=10)
        data_points.append({"a_labels": labels, "p_labels":[int(label[9:]) for label in predicted_labels[0]], "probabilities": predicted_labels[1]})
    for threshold in thresholds:
        data = {"type":"global",
                "threshold":threshold, 
                "total": len(test_data_local), 
                "unclass":0}
        for k in [*range(1,top_k+1,1)]:
            data.setdefault(k, 0)
        
        for d in data_points:
            labels = d["a_labels"]
            predicted = d["p_labels"]
            probabilities = d["probabilities"]
            possible_labels = [labels for labels, proba in zip(predicted, probabilities) if proba >= threshold]
            
            if len(possible_labels) == 0:
                data["unclass"] += 1
                continue
            for k in [*range(1,top_k+1,1)]:
                intersection = set(labels) & set(possible_labels[:k])
                if intersection:
                    data[k]+=1
                    
        for k in [*range(1,top_k+1,1)]:
            d = data["total"] - data["unclass"]
            if d != 0:
                data[k] /= data["total"] - data["unclass"]
            
        data["unclass"] /= data["total"]
        results.append(data)
    return pd.DataFrame(results)

def get_class_stats(test_data, model, ddc_classes_to_include, top_k=10, thresholds=[i/10 for i in [*range(0, 10, 1)]]):
    test_data_local = test_data.copy(deep=True)
    test_data_local["topic_id"] = test_data_local["topic_id"].apply(lambda topic: topic // 10  * 10)
    test_data_local = test_data_local.groupby(["page_uri","abstract"])["topic_id"].apply(set).apply(list).reset_index(name="topics")
    test_data_y = []
    predicted_y = []
    
    for test_entry in test_data_local.itertuples():
        abstract = (test_entry.abstract).replace("\n", " ")
        labels = test_entry.topics
        test_data_y.append(labels)
        
        predicted_labels = model.predict(abstract, k=10)
        predicted_labels =list(zip(predicted_labels[1], predicted_labels[0]))
        
        predicted_labels = [(weight, int(label[9:])) for weight, label in predicted_labels]
        predicted_y.append(predicted_labels)
        

    return svm_model_k_fold.get_class_stats(test_data_y, predicted_y, ddc_classes_to_include, top_k=top_k, thresholds=thresholds)
    
    
def fasttext_k_fold(corpus_dir, ddc_classes_to_include = [*range(0,1000,10)], ngrams=2, lr=0.1, dim=100, epochs=50, verbose=2, random_state=25, pretrained_vec_file="", early_break=False):
    models = []
    global_res = []
    splits = 5 if not early_break else 1
    for k_fold_it in [*range(splits)]:
        print(k_fold_it)
        print("Preparing trainset")
        train_test_split = lambda ddc_class, df: get_train_test_split(ddc_class, df, k_fold_it, splits=5, random_state=random_state)
        train_test_data = iterate_over_corpus(corpus_dir, train_test_split, ddc_classes_to_include)
        
        train_data_path = "D:/Informatikstudium/bachelor/stp/src/model_eval_scripts/fasttext_train.txt"
        train_data = pd.concat([i[0] for i in train_test_data]).drop_duplicates().reset_index(drop=True)
        preprocess_data(train_data, train_data_path)
        print("Training model")
        model = ""
        if pretrained_vec_file == "":
            model = fasttext.train_supervised(train_data_path, wordNgrams = ngrams, epoch=epochs, lr=lr, dim=dim, verbose=verbose)
        else:
            model = fasttext.train_supervised(train_data_path, wordNgrams = ngrams, epoch=epochs, lr=lr, dim=dim, verbose=verbose, pretrainedVectors=pretrained_vec_file)
        os.remove(train_data_path)
        
        print("Testing model")
        test_data = pd.concat([i[1] for i in train_test_data]).drop_duplicates().reset_index(drop=True)
        test_data_path = "D:/Informatikstudium/bachelor/stp/src/model_eval_scripts/fasttext_test.txt"
        preprocess_data(test_data, test_data_path)
        global_res.append((model.test(test_data_path), pd.concat([get_global_stats(test_data, model), get_class_stats(test_data, model, ddc_classes_to_include)])))
        os.remove(test_data_path)
        
        models.append(model)
    return global_res, models
        