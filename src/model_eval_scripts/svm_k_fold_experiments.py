# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:51:50 2023

@author: Adrian Kuhn
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from corpus_generation.corpus_gen_utils import load_obj, save_obj
from model.svm_model_k_fold import k_fold_svm
import pandas as pd
from model.k_fold_helper import df_to_latex_table
ddc_classes_to_include = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,570,580,590,610,630,650,670,680,710,720,730,740,780,790]
ddc_classes_to_include1 = [0,70,150,180,330,340,350,360,390,410,510,520,530,540,550,560,610,630,650,670,680,710,720,730,740,780,790]
#%% parameter config linear kernel
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_models.obj").replace("\\", "/"))
results_linear = []
params_c = [0.001, 0.01, 0.1, 1, 10]
for c in params_c:
    print(f"iteration c={c}")
    g_res, model = k_fold_svm(models, src_dir, svm_C=c, svm_type="linear", early_break=True)
    g_res = g_res[0]
    g_res["type"] = g_res["type"].apply(lambda x: "linear")
    g_res["svm_c"] = [c for i in [*range(len(g_res.index))]]
    g_res["svm_gamma"] = ["-" for i in [*range(len(g_res.index))]]
    g_res["svm_degree"] = ["-" for i in [*range(len(g_res.index))]]
    g_res["svm_coef0"] = ["-" for i in [*range(len(g_res.index))]]
    results_linear.append(g_res)

  
#%% parameter config rbf kernel
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_models.obj").replace("\\", "/"))
results_rbf = []
params_c = [0.001, 0.01, 0.1, 1, 10]
params_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
for c in params_c:
    for g in params_gamma:
        print(f"iteration c={c}, g={g}")
        g_res, model = k_fold_svm(models, src_dir, svm_C=c, svm_gamma=g, svm_type="rbf", early_break=True)
        g_res = g_res[0]
        g_res["type"] = g_res["type"].apply(lambda x: "rbf")
        g_res["svm_c"] = [c for i in [*range(len(g_res.index))]]
        g_res["svm_gamma"] = [g for i in [*range(len(g_res.index))]]
        g_res["svm_degree"] = ["-" for i in [*range(len(g_res.index))]]
        g_res["svm_coef0"] = ["-" for i in [*range(len(g_res.index))]]
        results_rbf.append(g_res)
#%% parameter config poly kernel
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_models.obj").replace("\\", "/"))
results_poly = []
params_c = [0.001, 0.01, 0.1, 1, 10]
params_degree = [2, 3, 4, 5]
params_coef0 = [0, 0.3, 0.6, 1]
for c in params_c:
    for degree in params_degree:
        for coef0 in params_coef0:
            print(f"iteration c={c}, degree={degree}, coef0={coef0}")
            g_res, model = k_fold_svm(models, src_dir, svm_C=c, svm_degree=degree, svm_coef0=coef0, svm_type="poly", early_break=True)
            g_res = g_res[0]
            g_res["type"] = g_res["type"].apply(lambda x: "poly")
            g_res["svm_c"] = [c for i in [*range(len(g_res.index))]]
            g_res["svm_gamma"] = ["-" for i in [*range(len(g_res.index))]]
            g_res["svm_degree"] = [degree for i in [*range(len(g_res.index))]]
            g_res["svm_coef0"] = [coef0 for i in [*range(len(g_res.index))]]
            results_poly.append(g_res)
#%% parameter config sigmoid kernel
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_models.obj").replace("\\", "/"))
results_sigmoid = []
params_c = [0.001, 0.01, 0.1, 1, 10]
params_coef0 = [0, 0.3, 0.6, 1]
for c in params_c:
    for coef0 in params_coef0:
        print(f"iteration c={c}, coef0={coef0}")
        g_res, model = k_fold_svm(models, src_dir, svm_C=c,  svm_coef0=coef0, svm_type="sigmoid", early_break=True)
        g_res = g_res[0]
        g_res["type"] = g_res["type"].apply(lambda x: "sigmoid")
        g_res["svm_c"] = [c for i in [*range(len(g_res.index))]]
        g_res["svm_gamma"] = ["-" for i in [*range(len(g_res.index))]]
        g_res["svm_degree"] = ["-" for i in [*range(len(g_res.index))]]
        g_res["svm_coef0"] = [coef0 for i in [*range(len(g_res.index))]]
        results_sigmoid.append(g_res)
#%% total config
results_config = pd.concat(results_linear + results_rbf + results_poly + results_sigmoid)
results_config.to_csv(SCRIPT_DIR + "/svm_config_data.csv", sep=";", index=False)
#%%
results_config = pd.read_csv(SCRIPT_DIR + "/svm_config_data.csv", sep=";")
table_cols = ['type', 'svm_c', 'svm_gamma', 'svm_degree', 'svm_coef0', '1', '2', '3']
s = df_to_latex_table(results_config.query("threshold==0"), table_cols, "llllllll", "caption", "label")
#%%
results_config_sorted = results_config.query("threshold==0").sort_values(by='1', ascending=False)
results_config_sorted = pd.concat(
    [results_config_sorted.query("type=='linear'")[:3],
     results_config_sorted.query("type=='rbf'")[:3],
     results_config_sorted.query("type=='poly'")[:3],
     results_config_sorted.query("type=='sigmoid'")[:3]])
table_cols = ['type', 'svm_c', 'svm_gamma', 'svm_degree', 'svm_coef0', '1', '2', '3']
s = df_to_latex_table(results_config_sorted, table_cols, "llllllll", "caption", "label")
#%% 1-1-1-1
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_models.obj").replace("\\", "/"))
g_res, model = k_fold_svm(models, src_dir, early_break=True, svm_C=0.1, train_chunk=5000)
#%%
res = pd.concat(g_res)
res = res.groupby(["type", "threshold"], as_index=False).mean()
save_obj(model, SCRIPT_DIR + "/svm_k_fold_1_1_1_1_models.obj")
res.to_csv(SCRIPT_DIR + "/svm_k_fold_1_1_1_1_results.csv")
#%%
res = pd.read_csv(SCRIPT_DIR + "/svm_k_fold_1_1_1_1_results.csv")
print(df_to_latex_table(res, ["threshold", "unclass", "1", "2", "3", "4", "5", "6"], "lllllllll", "", "tab:eval_matrix_svm_global"))
#%% 1-1-1-1 adjusted
path = "D:/Informatikstudium/bachelor/stp/src/model_eval_scripts/stm_k_fold_1_1_1_1_data.csv"
a = pd.read_csv(path, sep=";")
b = a.query("ranking_func=='ranking_c' and threshold==0 and ( `1` > 0.35  or `2` > 0.6) and type!='global'")
ddc_classes_to_include = set(b["type"].apply(int))
del a,b
#%%
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
#models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_models.obj").replace("\\", "/"))
res_svm, models_svm = k_fold_svm(models_stm, src_dir, ddc_classes_to_include,early_break=False, svm_C=0.1,  train_chunk=100000)
res_svm = pd.concat(res_svm).groupby(["type", "threshold"]).mean()

#models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_models.obj").replace("\\", "/"))
res_svm2, model_svm2 = k_fold_svm(models_stm2, src_dir, ddc_classes_to_include1,early_break=False, svm_C=0.1,  train_chunk=100000)
res_svm2 = pd.concat(res_svm2).groupby(["type", "threshold"]).mean()
#%%
save_obj(models, SCRIPT_DIR + "/svm_k_fold_1_1_1_1_adjusted.obj")
#%% 
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
models = load_obj((SCRIPT_DIR + "/stm_k_fold_1_1_1_1_adjusted_models.obj").replace("\\", "/"))
g_res2, models = k_fold_svm(models, src_dir, ddc_classes_to_include, early_break=False)
g_res2 = pd.concat(g_res2).groupby(["type", "threshold"]).mean()
save_obj(models, SCRIPT_DIR + "/svm_k_fold_1_1_1_1_adjusted+.obj")