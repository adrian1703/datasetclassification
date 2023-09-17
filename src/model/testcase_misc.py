# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:01:58 2023

@author: Adrian Kuhn
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.model_gen import construct_class_models, construct_stm_model
import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle
#%% Test case: Label weight threshold
src_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"
class_models, test_sets = construct_class_models(src_dir)
stm_model = construct_stm_model(src_dir, class_models)
#%%
from model.model_construction_helper import get_split, iterate_over_corpus
from model_eval_scripts.testcase_svm_ranking import get_train_test_split
train_test_split = lambda ddc_class, df: get_train_test_split(ddc_class, df, 0, random_state=25)
train_test_data = iterate_over_corpus(src_dir, train_test_split)
#%%
test_sets = [s[1] for s in train_test_data]
#%%
test = pd.concat(test_sets)
test_pages = test[["page_uri", "topic_id", "abstract_word_count","tokens"]].drop_duplicates().reset_index(drop=True)
test_pages["topic_id"] = test_pages["topic_id"].apply(lambda x: (x//10) *10)
df_expected = test_pages.groupby(['page_uri', "tokens"])["topic_id"].apply(list)
#%%
data = []
i = 0 
for (uri, tokens), topics in df_expected.iteritems():
    if i % 5000 == 0: print(i)
    scores = stm_model.calc_class_scores(tokens)
    percentiles = stm_model.calc_percentile_class_scores(scores)
    ddc_classes = stm_model.ddc_classes
    percentiles = list(zip(percentiles, ddc_classes))
    data.append([topics,percentiles])
    i += 1
df_eval = pd.DataFrame(data, columns=["actual", "calced_scores"])
#%% Eval sorted by class score
expected_cols = []
for row in df_eval.itertuples():
    exp = row.calced_scores
    exp = [(i[0][0], i[1]) for i in exp]
    expected_cols.append(sorted(exp, reverse=True))
df_eval["expected_2"] = expected_cols
#%% Eval sorted by not_class score
expected_cols = []
for row in df_eval.itertuples():
    exp = row.calced_scores
    exp = [(i[0][1], i[1]) for i in exp]
    expected_cols.append(sorted(exp, reverse=True))
df_eval["expected_3"] = expected_cols
#%% Eval sorted by class_score * not_class score
expected_cols = []
for row in df_eval.itertuples():
    exp = row.calced_scores
    exp = [(i[0][0] * i[0][1]/100, i[1]) for i in exp]
    expected_cols.append(sorted(exp, reverse=True))
df_eval["expected_4"] = expected_cols
#%% Eval matrix
cols = ["expected_2", "expected_3", "expected_4",]
data = []
for name in cols:
    for i in [*range(0,100,10)]:
        desc = f"{name}__{i}"
        temp_dict = {"describtion":desc}
        for x in [*range(1,10,1)]:
            pos = 0
            neg = 0
            unclass = 0
            for index, row in df_eval.iterrows():
                actual = row["actual"]
                expected = row[name][:x]
                expected = [j[1] for j in expected if j[0] > i]
                if not expected:
                    unclass += 1
                found = False
                for k in expected:
                    if k in actual:
                        found = True
                if found:
                    pos += 1
                else:
                    neg += 1
            temp_dict[f"{x}"] = pos / (pos + neg)
        temp_dict["unclass"] = unclass / (pos + neg + unclass)   
        data.append(temp_dict)         
eval_matrix = pd.DataFrame(data)

#%%

res3 = np.zeros(shape = (9,10))
for y in [*range(1, 10, 1)]:
    yy = y/10
    for x in [*range(1,10,1)]:
        pos = 0
        neg = 0
        unclass = 0
        for row in df_eval.itertuples():
            actual = row.actual
            expected = [(i[1], i[0]) for i in row.expected_2 if i[0]/100 >= yy]
            found = False
            for i,j in expected[:x]:
                if i in actual:
                    found = True
                    break
            if found:
                pos += 1
            elif expected:
                neg += 1
            else:
                unclass += 1
        res3[y-1][x-1] = pos/(pos+neg)
        res3[y-1][9] = unclass/(pos+neg+unclass)
        # print(f"--------------y = {y} x = {x}-----------------------")
        # print(pos/(pos+neg))
        # print(unclass/(pos+neg+unclass))
    # print(sum(avg_pos)/len(avg_pos))
    # print(sum(avg_neg)/len(avg_neg))
    
#%%
res3 = res2 - res
#%%

#%% svm test
testset = []
trainset =[]
corpus_dir = "D:/Informatikstudium/bachelor/stp/resources/corpus/grp_by_ddc_tokenized/1_1_1_1"

ddc_classes = [*range(0,1000,10)]
for ddc_class in ddc_classes:
    file_path = glob.glob(f"{corpus_dir}/{ddc_class}.csv")
    if not file_path:
        continue
    df = pd.read_csv(file_path[0], sep=";")
    df = df.drop_duplicates().reset_index(drop=True)
    df = shuffle(df, random_state=(1))
    split = (len(df.index) // 5) * 4
    trainset.append(df[:split]) 
    testset.append(df[split:])
#%% svm test
testset = pd.concat(testset)
trainset = pd.concat(trainset)
#%% svm test
trainset_x = [stm_model.calc_class_scores(row.tokens) for row in trainset.itertuples()]
testset_x = [stm_model.calc_class_scores(row.tokens) for row in testset.itertuples()]
#%% svm test
trainset_y = [stm_model.get_class_i(i//10 * 10)  for i in trainset[["topic_id"]].values.ravel()]
testset_y = [stm_model.get_class_i(i//10 * 10)   for i in testset[["topic_id"]].values.ravel()]
#%% svm test
from datetime import datetime
from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
#for c in [0.03,0.05, 0.1,0.5, 1, 3]:
    #     for g in [0.001,0.01, 0.1, 1, 10 , 100 , 1000]:
z = 75000
x = shuffle(trainset_x, random_state=25)[:z]
y = np.array(shuffle(trainset_y, random_state=25)[:z]).reshape(-1,1)
scaler_x = preprocessing.StandardScaler().fit(x,y)
x_scaled = scaler_x.transform(x)
t = datetime.now()
# linear_svc = svm.SVC(kernel='rbf', cache_size=2000, gamma=g, C=c, decision_function_shape='ovr')
linear_svc = svm.SVC(kernel='linear', cache_size=4000,  C=0.1, decision_function_shape='ovr')
linear_svc.fit(x_scaled, y.ravel())
print(datetime.now()-t)
#print("evaluating")
#%% svm test
xx = scaler_x.transform(testset_x)
predicted = linear_svc.predict(xx)
expected = testset_y
pos = []
neg = []
for i in [*range(len(predicted))]:
    p = predicted[i]
    e = expected[i]
    if p == e:
        pos.append(p)
    else:
        neg.append((e,p))
        # print(f"{[c,g,pos/(pos+neg)]}")
print(pos/(pos+neg))
#%% force directed graph
# source-targed-weight
import pickle
def get_force_direct_df(stm_model):
    s_t_w_list = []
    for ddc_class in stm_model.ddc_classes:
        index = stm_model.get_class_i(ddc_class)
        source = ddc_class
        weights = np.zeros(shape=(len(stm_model.ddc_classes), 1), dtype=float)
        for token, values in stm_model.model.items():
            for i in [*range(len(values))]:
                if i == index:
                    continue
                weights[i] += values[index] * values[i]
        for i in [*range(len(stm_model.ddc_classes))]:
            if i == index:
                continue
            target = stm_model.ddc_classes[i]
            s_t_w_list.append({
                "source":str(source),
                "target":str(target),
                "weight":weights[i]})
    return pd.DataFrame(s_t_w_list)
def save_obj(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
def load_obj(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
#%% force directed graph
weights_df = get_force_direct_df(stm_model)
#%%
weights_df.to_csv("D:/Informatikstudium/bachelor/stp/resources/stm_model-1-1-1-1.obj", sep=";", index=False)
#%% class approximation visual
#
#
#
#
#
#
#
#
#

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ddc_index = 0
for i in stm_model.ddc_classes:   
    # if ddc_index > 10 :
    #     break
    dcc_name = stm_model.ddc_classes[ddc_index]
    class_score_0 = [stm_model.calc_class_scores(row.tokens)[ddc_index] for row in test_pages.itertuples() if dcc_name == row.topic_id]
    class_score_other  = [stm_model.calc_class_scores(row.tokens)[ddc_index] for row in test_pages.itertuples()if dcc_name != row.topic_id]
    threshhold_train = stm_model.class_threshholds_pos[dcc_name]
    threshhold_train2 = stm_model.class_threshholds_neg[dcc_name]
    # class_score_other1 =  [row.class_scores for row in test_pages.itertuples()if dcc_name not in row.topic_id]
    pio.renderers.default='browser'
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x = list(threshhold_train.keys()),
        y = list(threshhold_train.values()),
        name="train approx pos",
        mode = "lines"
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x = list(threshhold_train2.keys()),
        y = list(threshhold_train2.values()),
        name="train approx neg",
        mode = "lines"
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x = [i / len(class_score_0) * 100 for i in [*range(len(class_score_0))]],
        y = sorted(class_score_0),
        name=f"test data {dcc_name}",
        mode = "lines",

        ), secondary_y=False)
    others = []
    
    fig.add_trace(go.Scatter(
        x = [i / len(class_score_other)  * 100 for i in [*range(len(class_score_other))]],
        y = sorted(class_score_other) ,
        name=f"test data other classes",
        mode = "lines",

        ), secondary_y=False)
    # fig.add_trace(go.Scatter(
    #     x = [i / len(class_score_0) * 100 for i in [*range(len(class_score_0))]],
    #     y = list(zip(*sorted(class_score_0)))[1],
    #     name=f"class {dcc_name}",
    #     mode = "markers",
    #     marker=dict(
    #         size=2,
    #     ),

    #     ), secondary_y=True)
    # fig.add_trace(go.Scatter(
    #     x = [i / len(class_score_other)  * 100 for i in [*range(len(class_score_other))]],
    #     y = list(zip(*sorted(class_score_other)))[1],
    #     name="other classes",
    #     mode = "markers",
    #     marker=dict(
    #         size=2,
    #     ),
    #     ), secondary_y=True)
    fig.show()
    ddc_index += 1
    
#%%
# evaluating
# [0.001, 0.001, 0.0676745970836531]
# evaluating
# [0.001, 0.01, 0.0676745970836531]
# evaluating
# [0.001, 0.1, 0.0676745970836531]
# evaluating
# [0.001, 1, 0.0676745970836531]
# evaluating
# [0.001, 10, 0.0676745970836531]
# evaluating
# [0.001, 100, 0.0676745970836531]
# evaluating
# [0.001, 1000, 0.0676745970836531]
# evaluating
# [0.01, 0.001, 0.0676745970836531]
# evaluating
# [0.01, 0.01, 0.07625479662317729]
# evaluating
# [0.01, 0.1, 0.0676745970836531]
# evaluating
# [0.01, 1, 0.0676745970836531]
# evaluating
# [0.01, 10, 0.0676745970836531]
# evaluating
# [0.01, 100, 0.0676745970836531]
# evaluating
# [0.01, 1000, 0.0676745970836531]
# evaluating
# [0.1, 0.001, 0.1214888718342287]
# evaluating
# [0.1, 0.01, 0.23183422870299308]
# evaluating
# [0.1, 0.1, 0.09823484267075978]
# evaluating
# [0.1, 1, 0.0676745970836531]
# evaluating
# [0.1, 10, 0.0676745970836531]
# evaluating
# [0.1, 100, 0.0676745970836531]
# evaluating
# [0.1, 1000, 0.0676745970836531]
# evaluating
# [1, 0.001, 0.28082885648503453]
# evaluating
# [1, 0.01, 0.35933998465080585]
# evaluating
# [1, 0.1, 0.26577129700690716]
# evaluating
# [1, 1, 0.07375287797390637]
# evaluating
# [1, 10, 0.07120491174213354]
# evaluating
# [1, 100, 0.07103607060629318]
# evaluating
# [1, 1000, 0.07089792785878742]
# evaluating
# [10, 0.001, 0.3756561780506523]
# evaluating
# [10, 0.01, 0.36655410590943976]
# evaluating
# [10, 0.1, 0.27177283192632384]
# evaluating
# [10, 1, 0.07513430544896393]
# evaluating
# [10, 10, 0.07128165771297007]
# evaluating
# [10, 100, 0.07106676899462779]
# evaluating
# [10, 1000, 0.07094397544128933]
# evaluating
# [100, 0.001, 0.3817805065234075]
# evaluating
# [100, 0.01, 0.3402762854950115]
# evaluating
# [100, 0.1, 0.2715886415963162]
# evaluating
# [100, 1, 0.07513430544896393]
# evaluating
# [100, 10, 0.07128165771297007]
# evaluating
# [100, 100, 0.07106676899462779]
# evaluating
# [100, 1000, 0.07094397544128933]
# evaluating
# [1000, 0.001, 0.36178050652340754]
# evaluating
# [1000, 0.01, 0.3381887950882579]
# evaluating
# [1000, 0.1, 0.2716039907904835]
# evaluating
# [1000, 1, 0.07514965464313124]
# evaluating
# [1000, 10, 0.07128165771297007]
# evaluating
# [1000, 100, 0.07106676899462779]
# evaluating
# [1000, 1000, 0.07094397544128933]
