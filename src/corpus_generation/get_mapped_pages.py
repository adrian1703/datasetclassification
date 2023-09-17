# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:54:37 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from model.model_construction_helper import  iterate_over_corpus

from model_eval_scripts.testcase_svm_ranking import get_train_test_split,remove_duplicates_test_data
from model.model_construction_helper import  iterate_over_corpus
import pandas as pd
#%%
srcs = ["D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_0_0_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/0_1_0_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/0_0_1_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_2",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_3",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_4",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_5",
        ]

def mapped_pages(ddc_class, df):
    return len(df[["page_uri"]].drop_duplicates().reset_index(drop=True).index)
for src_dir in srcs:
    result = iterate_over_corpus(src_dir, lambda ddc_class, df: mapped_pages(ddc_class, df))
    print( sum(result))
#%%

#%%