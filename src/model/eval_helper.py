# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:54:25 2023

@author: Adrian Kuhn
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sklearn.utils import shuffle
import pandas as pd
import model.model_construction_helper as helper
import model.model_gen as model_gen

# calculate ddc_scores for test-pages belonging to given ddc class in a given
# k fold iteration
# used with iterator_over_corpus to calculate scores for all test pages of a corpus
def eval_df_k_fold(ddc_class, df, k_fold_iter, stm_model, random_state=25):
    results = []
    local_df = df.drop_duplicates().reset_index(drop=True)
    local_df = shuffle(local_df, random_state=random_state)
    train_df, test_df = helper.get_split(k_fold_iter, local_df)
    for row in test_df.itertuples():
        uri = row.page_uri
        tokens = row.tokens
        scores = stm_model.calc_class_scores(tokens, convert_str_to_token=True)
        percentiles = stm_model.calc_percentile_class_scores(scores)
        classes = stm_model.ddc_classes
        results.append({
            "page_uri":uri, 
            "actual":ddc_class, 
            "scores":list(zip(percentiles, classes))
            })
    return pd.DataFrame(data=results)

# do a k_fold on a given corpus directory using percentile_func to evaluate 
#   score results
# return a list of dfs for each iteration and a list of constructed models
def k_fold(corpus_dir, percentile_funcs, splits=5, random_state=25, classes_to_exclude=[]):
    results = []   
    models = []
    for i in [*range(splits)]:
        print(i)
        construct_classes_func = lambda ddc_class, df: helper.construct_class_models_k_fold(ddc_class, df, k_fold_iter=i, random_state=random_state)
        class_models = helper.iterate_over_corpus(corpus_dir, construct_classes_func, classes_to_exclude)
        
        stm_model = model_gen.StmModel()
        stm_model.construct_model(class_models)
        assign_class_scores_func = lambda ddc_class, df: helper.assign_class_scores_stm_model_k_fold(ddc_class, df, k_fold_iter=i, stm_model=stm_model, random_state=random_state)
        helper.iterate_over_corpus(corpus_dir, assign_class_scores_func, classes_to_exclude)
        stm_model.calc_threshold_percentiles()
        
        eval_df_func = lambda ddc_class, df: eval_df_k_fold(ddc_class, df, k_fold_iter=i, stm_model=stm_model, random_state=random_state)
        page_results = helper.iterate_over_corpus(corpus_dir, eval_df_func, classes_to_exclude)
        
        eval_df = get_global_stats(merge_page_results(page_results), percentile_funcs)
        eval_df2 = get_class_stats(page_results, percentile_funcs)
        results.append(pd.concat([eval_df, eval_df2]))
        models.append(stm_model)
    return results, models

# Apply a percentile_function to the scores tuple ((a,b), g) -> (c, g)
# Sort the sorces by c descending an return the top k scores that are 
#   above the threshold
# scores:           list of ((a,b), g)
# percentile_func:  lambda ((a,b), g) -> (c,g)
# threshold:        number
# top_k:            number | how many top entries to fetch
def ranking_function(scores, percentile_func, threshold=0, top_k=10):
    local_scores = sorted([(percentile_func(i[0]),i[1]) for i in scores], reverse=True)
    local_scores = [i for i in local_scores[:top_k] if i[0] > threshold]
    return local_scores

# Merge page dataframes ["page_uri", "actual", "scores"] to one dataframe
#   "page_uri":str | uri of a given page
#   "actual": int | ddc class
#   "scores": list of ((a,b),g) |   a: int [0,..,99] | class score percentile
#                                   b: int [0,..,99] | not class_score_percentile
#                                   g: int | ddc_class
# Group entries by "page_uri" and "scores"
# Results to: 
#   "page_uri":str | uri of a given page
#   "actual": list of int | ddc classes according to mapping
#   "scores": list of ((a,b),g)
# Eliminates duplicate "page_uri" entries and groups ddc classes
    
def merge_page_results(page_results):
    complete_res = pd.concat(page_results)
    complete_res1 = complete_res[["page_uri", "actual"]].groupby("page_uri")["actual"].apply(list).reset_index(name="actual")
    complete_res2 = complete_res[["page_uri", "scores"]].groupby("page_uri", as_index=False).first()
    return pd.merge(complete_res1, complete_res2, on="page_uri")

# Use merged page results ["page_uri", "actual", "scores"] to evaluate
#   how many documents(page_uris) were labelled correctly
# ranking_funcs:    dict with {str: lambda ((a,b),g) -> (c,g)}
# top_k:            how many top ranked labels are included in evaluation
# thresholds:       list of thesholds used in ranking function application
# for each threshold and ranking_function construct an eval_matrix entry
# eval_matrix entry:
#       "type":             "global"
#       "threshold":        curren_threshold
#       "total":            total pages in merged_page_results
#       "unclass":          percentage of unclassed documents due to threshold
#       "ranking_function": ranking_func description
#       1:                  percent of correctly labelled docs by using the top 1 score entry
#       2:                  percent of correctly labelled docs by using the top 2 score entries
#       ...:
#       top_k:              percent of correctly labelled docs by using the top k score entry
def get_global_stats(merged_page_results, ranking_funcs, top_k=10, thresholds=[*range(0,100,10)]):
    results = []
    for threshold in thresholds:
        for ranking_func_name, ranking_func in ranking_funcs.items():
            data = {"type":"global", 
                    "threshold" : threshold, 
                    "total":len(merged_page_results.index),
                    "unclass" : 0, 
                    "ranking_func":ranking_func_name}
            for row in merged_page_results.itertuples():
                labels_actual = row.actual
                labels_expected = ranking_function(row.scores, ranking_func, threshold=threshold, top_k=top_k)
                if not labels_expected:
                    data["unclass"] += 1
                for i in [*range(1,top_k+1)]:
                    data.setdefault(i,0)
                    labels_i = {j[1] for j in labels_expected[:min(i,len(labels_expected))]}
                    intersetion = labels_i & set(labels_actual)
                    if intersetion:
                        data.setdefault(i,0)
                        data[i] += 1
            for i in [*range(1,top_k+1)]:
                data[i] = data[i] / (len(merged_page_results) - data["unclass"])
            data["unclass"] /= len(merged_page_results)
            results.append(data)
    return pd.DataFrame(results)

    
# Use page results ["page_uri", "actual", "scores"] to evaluate
#   how many documents(page_uris) were labelled correctly
# ranking_funcs:    dict with {str: lambda ((a,b),g) -> (c,g)}
# top_k:            how many top ranked labels are included in evaluation
# thresholds:       list of thesholds used in ranking function application
# for each threshold and ranking_function and ddc_class construct an eval_matrix entry
# eval_matrix entry:
#       "type":             ddc_class
#       "threshold":        curren_threshold
#       "total":            total pages in merged_page_results
#       "unclass":          percentage of unclassed documents due to threshold
#       "ranking_function": ranking_func description
#       1:                  percent of correctly labelled docs by using the top 1 score entry
#       2:                  percent of correctly labelled docs by using the top 2 score entries
#       ...:
#       top_k:              percent of correctly labelled docs by using the top k score entry
def get_class_stats(page_results, ranking_funcs, top_k=10, thresholds=[*range(0,100,10)]):
    results = []
    for page_df in page_results:
        ddc_class = page_df["actual"][0]  
        data = {threshold:{key:{} for key in ranking_funcs.keys()} for threshold in thresholds}
        for row in page_df.itertuples():
            for threshold in thresholds:          
                for ranking_func_name, ranking_func in ranking_funcs.items():
                    current_data= data[threshold][ranking_func_name]
                    current_data.setdefault("type", ddc_class)
                    current_data.setdefault("threshold", threshold)
                    current_data.setdefault("total",len(page_df.index))
                    current_data.setdefault("unclass" , 0)
                    current_data.setdefault("ranking_func",ranking_func_name)

                        
                    labels_expected = ranking_function(row.scores, ranking_func, threshold=threshold, top_k=top_k)
                    if not labels_expected:
                        current_data["unclass"] += 1  
                    for i in [*range(1,top_k+1)]:
                        current_data.setdefault(i,0)
                        labels_i = {j[1] for j in labels_expected[:min(i,len(labels_expected))]}
                        if ddc_class in labels_i:
                            current_data[i] += 1
        data_list = [all_values.values() for all_values in [th_values for th_values in data.values()]]
        data_list = [i for j in data_list for i in j]
        for data_entry in data_list:
            for i in [*range(1,top_k+1)]:
                if len(page_df) - data_entry["unclass"] == 0:
                    data_entry[i] = 0
                else:
                    data_entry[i] = data_entry[i] / (len(page_df) - data_entry["unclass"])
            data_entry["unclass"] /= len(page_df)
            results.append(data_entry)
    return pd.DataFrame(results)



def df_to_latex_table(df, cols, tabular_cols, caption, label):
    table_string = """
    \\begin{table}     
        \centering
        \setlength{\\tabcolsep}{6pt} % separator between columns (standard = 6pt)
        \\renewcommand{\\arraystretch}{1.25} % vertical stretch factor (standard = 1.0)
        \\begin{tabular}{@{}$tabular_cols@{}}
        \\toprule
        $header \\\\ 
        \midrule
        $cols   
        \\bottomrule
        \end{tabular}
        \caption{$caption}
        \label{$label}
    \end{table}
    """
    table_string= table_string.replace("$label", label)
    table_string= table_string.replace("$caption", caption)
    table_string= table_string.replace("$tabular_cols", tabular_cols)
    table_string= table_string.replace("$header", " & ".join([str(i) for i in cols]))
    
    data_str = []
    df_local= df[cols]
    for index, series in df_local.iterrows():
        col_data = [str(round((series[col]),4))  for col in cols]
        col_str = " & ".join(col_data) + "\\\\"
        data_str.append(col_str)
    data_str = " \n\t\t".join(data_str)
    
    table_string= table_string.replace("$cols", data_str)
    return table_string.replace("$tabular_cols", "asdf")

