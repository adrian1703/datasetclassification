# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:19:59 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.fasttext_k_fold import fasttext_k_fold
from corpus_generation.corpus_gen_utils import save_obj, load_obj
import pandas as pd
from model.k_fold_helper import df_to_latex_table
ddc_classes_to_include = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,570,580,590,610,630,650,670,680,710,720,730,740,780,790]
ddc_classes_to_include1 = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,610,630,650,670,680,710,720,730,740,780,790]

#%% configuration
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
dim = [50,100,300]
lr = [0.1, 0.3, 0.6]
epochs = [5, 25, 50]
n_gram = [2, 4, 6]
pretrained = [False, True]
g_res = []
for d in dim:
    for l in lr:
        for e in epochs:
            for n in n_gram:
                print([d,l, e,n])
                global_res, models =fasttext_k_fold(src_dir, ngrams=n, lr=l, dim=d, epochs=e, verbose=3, early_break=True)
                x = global_res[0][1].query("threshold==0")
                g_res.append( {"dim":d, "lr":l, "epochs":e, "ngram":n, "pretrained":False, "1":x[1][0], "2":x[2][0], "3":x[3][0]})
g_res = pd.DataFrame(g_res)
#%% configuration pretrained
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
vec_file = "D:/Informatikstudium/bachelor/wiki-news-300d-1M-subword.vec/wiki-news-300d-1M-subword.vec"
dim = [300]
lr = [0.1, 0.3, 0.6]
epochs = [5, 25, 50]
n_gram = [2]
pretrained = [False, True]
g_res = []
for d in dim:
    for l in lr:
        for e in epochs:
            for n in n_gram:
                print([d,l, e,n])
                global_res, models =fasttext_k_fold(src_dir, ngrams=n, lr=l, dim=d, epochs=e, verbose=3, early_break=True)
                x = global_res[0][1].query("threshold==0")
                g_res.append( {"dim":d, "lr":l, "epochs":e, "ngram":n, "pretrained":True, "1":x[1][0], "2":x[2][0], "3":x[3][0]})
g_res = pd.DataFrame(g_res)
#%%
# save_obj(g_res, SCRIPT_DIR + "/temp.data")
g_res2 = load_obj(SCRIPT_DIR + "/temp.data")
g_res2["pretrained"] = g_res2["pretrained"].apply(lambda x: "true" if x else "False")
table_cols = ['dim', 'lr', 'epochs', 'ngram', 'pretrained', '1', '2', '3']
s = df_to_latex_table(g_res2, table_cols, "llllllll", "caption", "label")
table_cols = ['dim', 'lr', 'epochs', 'ngram', 'pretrained', '1', '2', '3']
s = df_to_latex_table(g_res2.sort_values(by="1", ascending=False), table_cols, "llllllll", "caption", "label")
#%%
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
res_fasttext3, models_fasttext3 = fasttext_k_fold(src_dir, early_break=True, epochs=5, ngrams=2, dim=100, lr=0.6)
res_fasttext3 = pd.concat([i[1] for i in res_fasttext3]).groupby(["type", "threshold"], as_index=False).mean()
#%%
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
global_res, models = fasttext_k_fold(src_dir, ddc_classes_to_include=ddc_classes_to_include, early_break=False, epochs=5, ngrams=2, dim=100, lr=0.6)
global_res = pd.concat([i[1] for i in global_res]).groupby(["type", "threshold"], as_index=False).mean()


src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
global_res2, models = fasttext_k_fold(src_dir,ddc_classes_to_include=ddc_classes_to_include, early_break=False, epochs=50, ngrams=2, dim=100, lr=0.6)
global_res2 = pd.concat([i[1] for i in global_res2]).groupby(["type", "threshold"], as_index=False).mean()
#%%
s = df_to_latex_table(global_res2, ["threshold", "unclass", 1, 2, 3, 4, 5, 6], "lllllllll", "", "tab:eval_matrix_ranking_a_global")
#%%
save_obj(g_res, SCRIPT_DIR + "/global_res.data")
save_obj(g_res, SCRIPT_DIR + "/global_res2.data")
#%%
a = load_obj(SCRIPT_DIR + "/temp.data")
#%%
vec_file = "D:/Informatikstudium/bachelor/wiki-news-300d-1M-subword.vec/wiki-news-300d-1M-subword.vec"
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
global_res_pretrained, models = fasttext_k_fold(src_dir, early_break=True, epochs=5, ngrams=2, dim=300, lr=0.6, pretrained_vec_file=vec_file)
#global_res_pretrained = pd.concat([i[1] for i in global_res_pretrained]).groupby(["type", "threshold"], as_index=False).mean()

#%%
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
res_fasttext, models_fasttext = fasttext_k_fold(src_dir,ddc_classes_to_include, epochs=5, ngrams=2, dim=100, lr=0.6)
res_fasttext = pd.concat([i[1] for i in res_fasttext]).groupby(["type", "threshold"], as_index=False).mean()
#%%
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1"
res_fasttext2, models_fasttext2 = fasttext_k_fold(src_dir,ddc_classes_to_include1, epochs=5, ngrams=2, dim=100, lr=0.6)
res_fasttext2 = pd.concat([i[1] for i in res_fasttext2]).groupby(["type", "threshold"], as_index=False).mean()






