# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:32:26 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd

import os.path

from model.stm_model_k_fold import k_fold
from model.k_fold_helper import df_to_latex_table
import pickle
ddc_classes_to_include = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,570,580,590,610,630,650,670,680,710,720,730,740,780,790]
ddc_classes_to_include1 = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,610,630,650,670,680,710,720,730,740,780,790]

#%% 1-1-1-1 data experiment
percentile_funcs = {"ranking_a" : lambda a,b: a,
                    "ranking_b" : lambda a,b: b,
                    "ranking_c" : lambda a,b: a * b / 100}
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"

k_fold_results, models = k_fold(src_dir, percentile_funcs, ddc_classes=[*range(0,1000,10)])
res = pd.concat(k_fold_results)
res = res.groupby(["type", "threshold", "ranking_func"], as_index=False).mean()
#%% save to file
path = (SCRIPT_DIR).replace("\\", "/")
res.to_csv(path + "/stm_k_fold_1_1_1_1_data.csv", sep=";", index=False)
with open(path + "/stm_k_fold_1_1_1_1_models.obj", 'wb') as file:
        pickle.dump(models, file)
#%% 
df = pd.read_csv(path , sep=";")
#%%
df = df.query("ranking_func=='ranking_c' and threshold==0")
#%% ranking a 
df = res.query("ranking_func == 'ranking_a' and type == 'global'")
print(df_to_latex_table(df, ["threshold", "unclass", 1, 2, 3, 4, 5, 6], "lllllllll", "", "tab:eval_matrix_ranking_a_global"))
#%% ranking b
df = res.query("ranking_func == 'ranking_b' and type == 'global'")
print(df_to_latex_table(df, ["threshold", "unclass", 1, 2, 3, 4, 5, 6], "lllllllll", "", "tab:eval_matrix_ranking_a_global"))
#%% ranking c 
df = res.query("ranking_func == 'ranking_c' and type == 'global'")
print(df_to_latex_table(df, ["threshold", "unclass", 1, 2, 3, 4, 5, 6], "lllllllll", "", "tab:eval_matrix_ranking_a_global"))
# #%%
# df = res.query("threshold == 0 and type != 'global'")
#%% adjusted corpus 
path = "D:/Informatikstudium/bachelor/stp/src/model_eval_scripts/stm_k_fold_1_1_1_1_data.csv"
a = pd.read_csv(path, sep=";")
b = a.query("ranking_func=='ranking_c' and threshold==0 and ( `1` > 0.4  or `2` > 0.6) and type!='global'")
ddc_classes_to_include = set(b["type"].apply(int))
#%%
print(sorted(ddc_classes_to_include))
#%%
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
res_stm, models_stm = k_fold(src_dir, {"ranking_c" : lambda a,b: a * b / 100}, ddc_classes=ddc_classes_to_include, early_break=False)
res_stm = pd.concat(res_stm).groupby(["type", "threshold", "ranking_func"], as_index=False).mean()

res_stm2, models_stm2 = k_fold(src_dir, {"ranking_c" : lambda a,b: a * b / 100}, ddc_classes=ddc_classes_to_include1, early_break=False)
res_stm2 = pd.concat(res_stm2).groupby(["type", "threshold", "ranking_func"], as_index=False).mean()
#%%
res = res2.query("threshold==0")
#%%
res2.to_csv(SCRIPT_DIR + "/stm_k_fold_1_1_1_1_adjusted_data.csv", sep=";", index=False)
with open(SCRIPT_DIR + "/stm_k_fold_1_1_1_1_adjusted_models.obj", 'wb') as file:
        pickle.dump(models2, file)
