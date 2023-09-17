# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:22:02 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from model.fasttext_k_fold import get_train_test_split
from model.k_fold_helper import get_split, iterate_over_corpus
import pandas as pd
#%%

def get_corpus_stats(srcs):
    results = []
    for src in srcs:
        train_test_split = lambda ddc_class, df: get_train_test_split(ddc_class, df, 0, splits=5, random_state=42)
        train_test_data = iterate_over_corpus(src, train_test_split, [*range(0,1000,10)])
        all_dfs = []
        for i,j in train_test_data:
            all_dfs.append(i)
            all_dfs.append(j)
        corpus_df = pd.concat(all_dfs)
        corpus_df["topic_id"] = corpus_df["topic_id"].apply(lambda x: int(x) // 10 * 10 )
        corpus_df = corpus_df.drop_duplicates().reset_index(drop=True)
        corpus_df_no_dup = corpus_df[["page_uri"]]
        corpus_df_no_dup = corpus_df_no_dup.drop_duplicates().reset_index(drop=True)   
        results.append({"src": src, "total_pages":len(corpus_df.index), "unique_pages":len(corpus_df_no_dup.index), "percentage": len(corpus_df_no_dup.index)/len(corpus_df.index)})
    return pd.DataFrame(results)
#%%
srcs = ["D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_0_0_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/0_1_0_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/0_0_1_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_0",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_1",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_2",
        "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_3",
        # "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_4",
        # "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc/1_1_1_5",
        ]
result = get_corpus_stats(srcs)
#%%
print(result
      )