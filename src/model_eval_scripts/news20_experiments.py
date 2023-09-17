# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:40:53 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from corpus_generation.corpus_gen_utils import save_obj, load_obj
from glob import glob,iglob
from pprint import pprint
import pandas as pd
import fasttext
import model.stm_model_k_fold as stm
import model.svm_model_k_fold as svm
import model.fasttext_k_fold as ft
from collections import Counter
from model.k_fold_helper import df_to_latex_table
ddc_classes_to_include = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,570,580,590,610,630,650,670,680,710,720,730,740,780,790]
#%%
path = "D:/Informatikstudium/bachelor/20news-18828/20news-18828"
categories = glob(path+ "/*")
cats = {
        "comp" : categories[1:6],
        "rec" : categories[7:11],
        "misc" : [categories[6]],
        "politics" : categories[16:19],
        "science" : categories[11:15],
        "religion" : [categories[0], categories[19], categories[15]]
        }
categories = cats
del path, cats
#%% generate models | load 
# stm_models = load_obj(SCRIPT_DIR + "/stm_k_fold_1_1_1_1_adjusted_models.obj")
# svm_models = load_obj(SCRIPT_DIR + "/svm_k_fold_1_1_1_1_adjusted+.obj")
# ft_model = fasttext.load_model(SCRIPT_DIR + "/fasttext_k_fold_model_adjusted.bin")
# stm
print("gen stm model")
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
res_stm, stm_models = stm.k_fold(src_dir, {"ranking_c" : lambda a,b: a * b / 100}, ddc_classes=ddc_classes_to_include, early_break=True)
stm_model = stm_models[0]
print("gen svm extension")
res_svm, models_svm = svm.k_fold_svm(stm_models, src_dir, ddc_classes_to_include,early_break=False, svm_C=0.1,  train_chunk=30000)
print("gen fasttext")
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
res_fasttext, models_fasttext = ft.fasttext_k_fold(src_dir,ddc_classes_to_include, epochs=5, ngrams=2, dim=100, lr=0.6,early_break=True)

stm_model = stm_models[0]
svm_model = models_svm[0][0]
svm_scaler = models_svm[0][1]
ft_model = models_fasttext[0]
del res_fasttext, res_stm, res_svm, stm_models, models_fasttext, models_svm , src_dir
#%%

#%%
def calc_news20(model, k=1, threshold=0):
    results_dict = {}
    for cat, folders in categories.items():
        results_dict[cat] = []
        
        for folder in folders:
            files = glob(folder + "/*")
            for file_path in files:
                with open(file_path, "r") as file: 
                    s = file.read()
                    #####
                    if model == "ft":
                        cat_predicted = ft_model.predict(s.replace("\n", ""), k=k,threshold=threshold)
                        for entry in cat_predicted[0]:
                            results_dict[cat].append(int(entry[9:]))
                    #####
                    if model == "stm":
                        cat_predicted = stm_model.predict_doc(s, top_k=k,ranking_threshold=threshold*100)
                        for entry in cat_predicted[:k]:
                            results_dict[cat].append(entry[1])
                    ####
                    if model == "svm":
                        scores = stm_model.get_class_scores_doc(s)
                        scores_scaled = svm_scaler.transform(scores.reshape(1, -1))
                        res = svm_model.predict_proba(scores_scaled)
                        cat_predicted = [sorted(list(zip(weight, svm_model.classes_)), reverse=True)[:k]for weight in res ]
                        for entry in cat_predicted[0]:
                            if entry[0] < threshold: continue
                            results_dict[cat].append(entry[1])
                    ###
    return results_dict
#%%
def collect_test_data():
    res = []
    res_pd = []
    for model in ["stm", "svm", "ft"]:
        for th in [0, 0.3, 0.6, 0.9]:
            data = calc_news20(model, 1, th)
            res.append({"model":model, "th":th, "data":data})
    for entry in res:
        for cat, values in entry["data"].items():
            agg_values = pd.Series(values).value_counts()[:5]
            entry_dict = {"cat":cat, "model":entry["model"], "th":entry["th"]}
            for i in [*range(min(len(agg_values), 5))]:
                
                entry_dict[str(i+1)] =  agg_values[agg_values.index[i]]
                entry_dict[str(i+1) + "d"] =  agg_values.index[i]
            res_pd.append(entry_dict)
    return pd.DataFrame(res_pd)
#%%
def over_view ():
    res = []
    res_pd = []
    for model in ["stm", "svm", "ft"]:
        for th in [0, 0.3, 0.6, 0.9]:
            data = calc_news20(model, 1, th)
            res.append({"model":model, "th":th, "data":data})
    for entry in res:
        for cat, values in entry["data"].items():
            agg_values = pd.Series(values).value_counts()[:5]
            entry_dict = {"cat":cat, "model":entry["model"], "th":entry["th"], "total":len(values)}
            for i in [*range(min(len(agg_values), 5))]:
                
                entry_dict[str(i+1)] =  agg_values[agg_values.index[i]]
                entry_dict[str(i+1) + "d"] =  agg_values.index[i]
            res_pd.append(entry_dict)
    return pd.DataFrame(res_pd)
def detail_view(models, ths= [0, 0.3, 0.6, 0.9]):
    totals={"comp":4881,
    "rec":3977,
    "science":3949,
    "politics":2625,
    "religion":2424,
    "misc":972}
    ddc={"comp":[0,530],
    "rec":[790],
    "science":[530, 600, 610],
    "politics":[330,340,360],
    "religion":[],
    "misc":[]}
    res = []
    res_pd = []
    for model in models:
        print(model)
        for th in ths:
            data = calc_news20(model, 1, th)
            res.append({"model":model, "th":th, "data":data})
    for entry in res:
        for cat, values in entry["data"].items():
            pos = 0
            for i in ddc[cat]:
                pos += values.count(i)
            total = len(values)
            neg = total - pos
            if total !=0:
                pos = pos/total
                neg = neg/total
            unclass =  (totals[cat] - total) / totals[cat]
             
            entry_dict = {"cat":cat, "model":entry["model"], "th":entry["th"], "total":len(values), "unclass":unclass, "pos":pos, "neg":neg}
            res_pd.append(entry_dict)
    return pd.DataFrame(res_pd)

def classify_dataset(filenames, model='ft', threshold=0.0, k=1):
    res = []
    i = 0
    for filename in filenames:
        if not os.path.isfile(filename): continue
        with open(filename, "r") as file: 
            i+=1
            s = file.read()
            #####
            if model == "ft":
                cat_predicted = ft_model.predict(s.replace("\n", ""), k=k,threshold=threshold)
                for entry in cat_predicted[0]:
                    res.append(int(entry[9:]))
            #####
            if model == "stm":
                cat_predicted = stm_model.predict_doc(s, top_k=k,ranking_threshold=threshold*100)
                for entry in cat_predicted[:k]:
                    res.append(entry[1])
            ####
            if model == "svm":
                scores = stm_model.get_class_scores_doc(s)
                scores_scaled = svm_scaler.transform(scores.reshape(1, -1))
                res_svm = svm_model.predict_proba(scores_scaled)
                cat_predicted = [sorted(list(zip(weight, svm_model.classes_)), reverse=True)[:k]for weight in res_svm ]
                for entry in cat_predicted[0]:
                    if entry[0] < threshold: continue
                    res.append(entry[1])
            ###
    total = len(res)
    if total == 0: return 0, 1
    c = Counter(res)
    c = [(ele, count/total) for ele, count in c.most_common(10)]
    return c, (i-len(res))/i
#%% Agg 20 newsgroups stats for categories
b = detail_view(["stm","ft", "svm"], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#%%
b.sort_values(by=["cat", "model"])
s = df_to_latex_table(b.sort_values(by=["cat", "model"]).query("cat not in ['religion', 'misc'] and model in ['ft', 'svm'] and th in [0, 0.3, 0.6]"), ["cat", "model", "th", "total", "unclass", "pos", "neg"], "", "caption", "label")
#%% classify news 20 in total
path = "D:/Informatikstudium/bachelor/20news-18828/20news-18828"
filenames = glob(path + '/**/*', recursive = True)
filenames = [i.replace('\\', '/') for i in filenames]
a = []
for th in [0,0.1,0.2,0.3,0.4,0.5,0.6]:
    for model in ["ft", "svm", "stm"]:
        data, unclass = classify_dataset(filenames, model, th)
        b = {"model":model, "th":th, "unclass":unclass}
        
        for i in [*range(len(data))]:
            b.setdefault(i+1,0)
            b[i+1]= (data[i][0], round(data[i][1],3))
        a.append(b)    
a = pd.DataFrame(a)
#%%
#%%

ft_k_0  = calc_news20("ft", k=1, threshold=0)
ft_k_30 = calc_news20("ft", k=1, threshold=0.3)
ft_k_60 = calc_news20("ft", k=1, threshold=0.6)
ft_k_90 = calc_news20("ft", k=1, threshold=0.9)
#%%
svm_k_0  = calc_news20("svm", k=1, threshold=0)
svm_k_30 = calc_news20("svm", k=1, threshold=0.3)
svm_k_60 = calc_news20("svm", k=1, threshold=0.6)
svm_k_90 = calc_news20("svm", k=1, threshold=0.9)
#%%
stm_k_0  = calc_news20("stm", k=1, threshold=0)
stm_k_30 = calc_news20("stm", k=1, threshold=0.3)
stm_k_60 = calc_news20("stm", k=1, threshold=0.6)
stm_k_90 = calc_news20("stm", k=1, threshold=0.9)
#%%
from collections import Counter
Counter(ft_k_0['comp'])

a = pd.Series(svm_k_0['rec']).value_counts()[:5]
#%%
not os.path.isfile(filenames[30])
