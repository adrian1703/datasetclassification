# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:34:33 2023

@author: Adrian Kuhn
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import os.path
import glob
import math
from sklearn.utils import shuffle


# Iterates over corpus and executes a lambda function with 
# lambda ddc_class, df: result
# appending the results and returning them 
def iterate_over_corpus(src_folder, func, ddc_classes):
    result = []
    for ddc_class in ddc_classes:
        file_path = glob.glob(f"{src_folder}/{ddc_class}.csv")
        if not file_path:
            continue
        df = pd.read_csv(file_path[0], sep=";")
        
        result.append(func(ddc_class, df))
    return result

# Splits a dataframe into test and train set
# Splits depend on the k-fold iteration
# For sequential use of training
def get_split(iteration, df, splits=5, drop_duplicates_df=True, shuffle_df=True, random_state=25):
    local_df = df
    if drop_duplicates_df:
        local_df = df.drop_duplicates().reset_index(drop=True)
    if shuffle_df:
        local_df = shuffle(local_df, random_state=random_state)
    split_size = len(local_df.index) / splits
    indicies = [math.floor(split_size * i) for i in [*range(splits + 1)]]
    chunks = [local_df[indicies[i] : indicies[i + 1]] for i in [*range(splits)]]
    test = chunks.pop(iteration)
    train = pd.concat(chunks)
    return train, test

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
        col_data = [str(round((series[col]),4)) if not isinstance(series[col], str) else series[col] for col in cols]
        col_str = " & ".join(col_data) + "\\\\"
        data_str.append(col_str)
    data_str = " \n".join(data_str)
    
    table_string= table_string.replace("$cols", data_str)
    return table_string.replace("$tabular_cols", "asdf")