# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:21:02 2023

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
