# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:00:05 2023

@author: Adrian Kuhn
"""



from dash import Dash, dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def get_ddc_id_name_list (path_ddc_classes):
    f = open(path_ddc_classes)
    s = f.read()
    sSplit = s.split("\n")
    return [[i, sSplit[i]] for i in range(len(sSplit))]
resource_path = "D:/Informatikstudium/bachelor/stp/resources/"
ddc_classes = get_ddc_id_name_list(resource_path + "dewey_classes.txt")
ddc_classes = [i for i in ddc_classes if i[1] not in ["[Unassigned]", "(Optional number)"]]
ddc_classes_t100 = [i for i in ddc_classes if i[0] % 10 == 0]
mapping_files = {"inital_mapping" : "ddc_wikicat_mapping.csv", 
                 "manual_extension" : "ddc_wikicat_mapping_manual_extension.csv", 
                 "outline_extension" : "ddc_wikicat_mapping_outline_extension.csv",
                 "complete_0" : "ddc_wikicat_mapping_complete_0.csv", 
                 "complete_1" : "ddc_wikicat_mapping_complete_1.csv",
                 "complete_2" : "ddc_wikicat_mapping_complete_2.csv",
                 "complete_3" : "ddc_wikicat_mapping_complete_3.csv",
                 "complete_4" : "ddc_wikicat_mapping_complete_4.csv", 
                 "complete_5" : "ddc_wikicat_mapping_complete_5.csv"}
#%%
dfs = {key:pd.read_csv(resource_path + value, sep=";") for key, value in mapping_files.items()}
#dfs["inital_manual_mapping"] = pd.concat([dfs["inital_mapping"], dfs["manual_extension"]])
#%% map topic_id xxx to xx0
for name, df in dfs.items():
    df["topic_id"] = df["topic_id"].apply(lambda x: (x // 10) * 10)
    dfs[name] = df.drop_duplicates().reset_index(drop=True)
#%% basic metrics
def get_metrics(df):
    metric = {}
    metric["total_cats_mapped"] = len(df.index)
    classes = df["topic_id"]
    classes = set(classes.values)
    metric["mapped_classes"] = len(classes)
    cats_per_ddc = df.groupby(["topic_id"], as_index=False)\
                        .agg(count=pd.NamedAgg(column="uri", aggfunc=lambda x: len(x)))
    metric["cats_per_class"] = {row.topic_id : row.count for row in cats_per_ddc.itertuples()}
    return metric
metrics = {}
for desc, df in dfs.items():
    metrics[desc] = get_metrics(df)
    
#%% get table string for cats
table = ""
cats = sorted(list(set(pd.concat(dfs.values())["topic_id"].values)))
for cat in cats:
    table += f"{cat}  &"
    for metric in metrics.values():
        value = 0
        if cat in metric['cats_per_class'].keys():
            value = metric['cats_per_class'][cat]
        table += f"  {value}  &"
    table = table[:-1]
    table += " \\\\ \n"
#%%
ddc_mapping = pd.read_csv(resource_path + "ddc_wikicat_mapping.csv", sep=";")

mapped_classes = []

df_init = pd.read_csv(resource_path + mapping_files["inital_mapping"], sep=";")
df_step_1 = pd.concat([df_init, pd.read_csv(resource_path + mapping_files["manual_extension"], sep=";")])
df_step_2 = pd.concat([df_step_1, pd.read_csv(resource_path + mapping_files["outline_extension"], sep=";")])
df_step_3 = pd.read_csv(resource_path + mapping_files["complete_1"], sep=";")
data = {
        "inital_mapping" : df_init,
        "mapping_with_manual_ext" : df_step_1,
        "mapping_with_manual_ext_and_outline_ext" : df_step_2,
        "mapping_complete_and_depth_ext_1" : pd.read_csv(resource_path + mapping_files["complete_1"], sep=";"),
        "mapping_complete_and_depth_ext_2" : pd.read_csv(resource_path + mapping_files["complete_2"], sep=";"),
        "mapping_complete_and_depth_ext_3" : pd.read_csv(resource_path + mapping_files["complete_3"], sep=";"),
        "mapping_complete_and_depth_ext_4" : pd.read_csv(resource_path + mapping_files["complete_4"], sep=";"),
        "mapping_complete_and_depth_ext_5" : pd.read_csv(resource_path + mapping_files["complete_5"], sep=";"),
        
        }
#%%
for name, df in data.items():
    dclasses = list(dict.fromkeys(df["topic_id"]))
    dclasses_t100 = list(dict.fromkeys([i // 10 for i in dclasses]))
    mapped_classes.append([name, len(dclasses), len(dclasses_t100)])


mapped_classes = pd.DataFrame(data=mapped_classes, columns=["name", "distinct_classes", "distinct_classes_t100"])
mapped_classes_fig = px.line(mapped_classes, x="name", y="distinct_classes", markers=True,
                             title=f"Number of distinct Top 1000 DDC classes mapped to Wikipedia Categories (max={len(ddc_classes)})")
mapped_classes_fig_t100 = px.line(mapped_classes, x="name", y="distinct_classes_t100", markers=True,
                             title=f"Number of distinct Top 100 DDC classes mapped to Wikipedia Categories (max={len(ddc_classes_t100)})")
mapped_classes_fig.update_layout(yaxis_range=[0,len(ddc_classes)])
mapped_classes_fig_t100.update_layout(yaxis_range=[0,len(ddc_classes_t100)])
#%%
mapped_categories = []
for name, df in data.items():
    count = len(df.index)
    mapped_categories.append([name, count])
mapped_categories = pd.DataFrame(data=mapped_categories, columns=["name", "number_of_categories"])
mapped_categories_fig = px.line(mapped_categories, x="name", y="number_of_categories", log_y=True, markers=True,
                                title="Number of categories mapped to DDC classes")

#%%
import plotly.io as pio
pio.renderers.default='browser'
df = data["mapping_complete_and_depth_ext_5"]
class_dict = {i[0]:i[1]for i in ddc_classes}
data_radar = []
for key,value in class_dict.items():
    if key % 10 == 0:
        count = 0
        for i in [*range(10)]:
            count += len(df[(df.topic_id==(key+i))].index)
        if count == 0:continue
        data_radar.append([str(key) + ":" + value, count])
data_radar = pd.DataFrame(data=data_radar, columns=["name","count"]) 
df = data["mapping_complete_and_depth_ext_3"]
class_dict = {i[0]:i[1]for i in ddc_classes}
data_radar2 = []
for key,value in class_dict.items():
    if key % 10 == 0:
        count = 0
        for i in [*range(10)]:
            count += len(df[(df.topic_id==(key+i))].index)
        if count == 0:continue
        data_radar2.append([str(key) + ":" + value, count])
data_radar2 = pd.DataFrame(data=data_radar2, columns=["name","count"])   
#%%
radar_fig = px.line_polar(data_radar, r="count", theta="name", line_close=True)
radar_fig.update_polars(radialaxis={"type":"log"})
radar_fig.add_scatterpolar(r=data_radar2["count"])
radar_fig.update_traces(fill="toself")

radar_fig.show()
#%%
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
def get_radar_df(source, ddc_classes):
    df = source
    class_dict = {i[0]:i[1]for i in ddc_classes}
    data_radar = []
    for key,value in class_dict.items():
        if key % 10 == 0:
            count = 0
            for i in [*range(10)]:
                count += len(df[(df.topic_id==(key+i))].index)
            #if count == 0:continue
            data_radar.append([str(key) + ":" + value, count])
    return pd.DataFrame(data=data_radar, columns=["name","count"]) 
def get_radar_fig(data_df):
    fig = go.Figure()
    for df in data_df:
        fig.add_trace(go.Scatterpolar(
          r=df[1]["count"],
          theta=df[1]["name"],
          fill='toself',
          name=df[0]
            ))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          type="log"
        ),
      ),
      showlegend=True
    )
    return fig
data_radar = [
        ["complete_0", data["mapping_with_manual_ext_and_outline_ext"]],
        ["complete_1", data["mapping_complete_and_depth_ext_1"]],
        ["complete_2", data["mapping_complete_and_depth_ext_2"]],
        ["complete_3", data["mapping_complete_and_depth_ext_3"]],
        ["complete_4", data["mapping_complete_and_depth_ext_4"]],
        ["complete_5", data["mapping_complete_and_depth_ext_5"]]
        ]
data_radar = [[i[0],get_radar_df(i[1],ddc_classes)] for i in data_radar]
fig = get_radar_fig(data_radar)

fig.show()
#%%
mapping1 = data["mapping_complete_and_depth_ext_2"]
mapping2 = data["mapping_complete_and_depth_ext_3"]
mapping1_cat = [list(i) for i in list(mapping1.values)]
mapping2_cat = [list(i) for i in list(mapping2.values)]
diff = []
for i in mapping1_cat:
    if i not in mapping2_cat:
        diff.append(i)
#%%
d = []
for name, df in data.items():
    cats = list(dict.fromkeys(df["uri"]))
    print(name + " " + str(len(cats)))
    d.append([name, cats])
for i in [*range(len(d)-1)]:
    first = d[i][1]
    second = d[i+1][1]
    
    diff = list(dict.fromkeys(first + second))
    print(d[i][0] + " vs " + d[i+1][0] + " " + str(len(diff) - len(second)))
#%%
df1 = pd.read_csv(resource_path + "/ddc_wikicat_mapping_complete_5.csv", sep=";")
df2 = pd.read_csv(resource_path + "/ddc_wikicat_mapping_complete_5+.csv", sep=";")
diff = set(df1["uri"]) ^ set(df2["uri"])