# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:40:43 2023

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
radar_fig = px.line_polar(data_radar, r="count", theta="name", line_close=True)
radar_fig.update_polars(radialaxis={"type":"log"})
#%%
app = Dash("Corpus Analyse")

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Topic Coverage", children=[
            dcc.Graph(figure=mapped_classes_fig),
            dcc.Graph(figure=mapped_classes_fig_t100)]),
        dcc.Tab(label="Mapped categories", children=[
            dcc.Graph(figure=mapped_categories_fig),
            dcc.Graph(figure=radar_fig)
            ])
        ])
    #html.H4("DCC classes with a category mapping"),
   
    ])

app.run_server(debug=True, port=8051, use_reloader=True)