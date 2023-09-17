# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:32:14 2023

@author: Adrian Kuhn
"""

#%% 
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import gc
from datetime import datetime
from dotmap import DotMap
import glob
import os.path

from corpus_generation.corpus_gen_utils import get_ddc_id_name_list, query_dbpedia, WikiCat, WikiPage
from corpus_generation.corpus_gen_category_mapping_parse_outline import get_wikicat_from_dewey_outline
#%%

resources_path = "D:/Informatikstudium/bachelor/stp/resources/"
#resources_path = "C:/bachelor/stp/resources/"

def query_for_wikicats(possible_cats, lang='en'):
    params = []
    params.append(['$lang', 'en'])
    
    filter_str = "FILTER ( $entry )."
    for cat in possible_cats:
        filter_str = filter_str.replace('$entry', f'?label = "{cat}"@$lang || $entry')
    filter_str = filter_str.replace('|| $entry', '')
    
    query = """   
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?subject ?label

    WHERE { 
       ?subject rdfs:label ?label 
    FILTER (langMatches(lang(?label), "$lang")) .
    $filter
    FILTER (regex(?subject, "^http://dbpedia.org/resource/Category:*"))
    } LIMIT 100
    """.replace('$filter', filter_str)
    return query_dbpedia(query, params)

def extend_wikicats(cats_to_extend, cat_depth='1'):
    params = [['$cat_depth', cat_depth]]
    if(len(cats_to_extend) < 1): return
    union_str = "$entry"
    for cat in cats_to_extend:
        union_str = union_str.replace('$entry', "{?subject skos:broader{$cat_depth} <category> .} UNION $entry".replace('category', cat))
    union_str = union_str.replace(' UNION $entry', '')
    query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?subject WHERE {
      $union_str
    }
    """
    query = query.replace('$union_str', union_str)
    return query_dbpedia(query, params)

def get_wiki_pages_and_abstracts_for_wikicats(wikicats, lang='en'):    
    filter_str ="FILTER ( $entry )"
    for cat in wikicats:
        filter_str = filter_str.replace('$entry', f'?o = <{cat}> || $entry')
    filter_str = filter_str.replace('|| $entry', '')    
    params = []
    params.append(['$lang', lang])
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT DISTINCT ?sub ?o ?page ?abstract ?language WHERE {
      ?sub dct:subject ?o .
      ?sub foaf:isPrimaryTopicOf ?page .
      ?sub dbo:abstract ?abstract .
      BIND ( lang(?abstract) AS ?language ) 
      FILTER (lang(?abstract) = 'en')
      $filter_str
    } 
    """
    query = query.replace('$filter_str', filter_str)
    return query_dbpedia(query, params)


def get_exact_match_wikicats():
    ddc_id_name = get_ddc_id_name_list(resources_path + "dewey_classes.txt")
    exact_match_cats = []
    for i in ddc_id_name:
        name = i[1]
        if name == '[Unassigned]': continue
        names = name.split(',')
        names = [x.lstrip(' ') for x in names]
        
        res = DotMap(query_for_wikicats(names))
        for binding in res.results.bindings:
            exact_match_cats.append(WikiCat(name=binding.subject.value.split(':')[-1], uri=binding.subject.value, topic_id=i[0]))
    return pd.DataFrame(data=exact_match_cats)
#%% Parse initial 
cats_inital = get_exact_match_wikicats()      
cats_to_remove = pd.read_csv(resources_path + "ddc_wikicat_mapping_to_remove.csv", sep=";")
for cat in cats_to_remove.values:
    cats_inital = cats_inital[(cats_inital["uri"]!=cat[2])]  
cats_inital.to_csv(resources_path + "ddc_wikicat_mapping.csv", index=False, sep=";")
#%% Parse outline cats
cats_outline = get_wikicat_from_dewey_outline()
cats_outline.to_csv(resources_path + "ddc_wikicat_mapping_outline_extension.csv", index=False, sep=";")
#%% inital + manual extension
cats_inital = pd.read_csv(resources_path + "ddc_wikicat_mapping.csv", sep=";")
cats_manual_ext = pd.read_csv(resources_path + "ddc_wikicat_mapping_manual_extension.csv", sep=";")
cats_complete = (pd.concat([cats_inital, cats_manual_ext])) \
                    .sort_values(by="topic_id") \
                    .drop_duplicates() \
                    .reset_index(drop=True)
cats_complete.to_csv(resources_path + "ddc_wikicat_mapping_inital_manual_0.csv", index=False, sep=";")
#%% inital + outline extension
cats_inital = pd.read_csv(resources_path + "ddc_wikicat_mapping.csv", sep=";")
cats_outline_ext = pd.read_csv(resources_path + "ddc_wikicat_mapping_outline_extension.csv", sep=";")
cats_complete = (pd.concat([cats_inital, cats_outline_ext])) \
                    .sort_values(by="topic_id") \
                    .drop_duplicates() \
                    .reset_index(drop=True)
cats_complete.to_csv(resources_path + "ddc_wikicat_mapping_inital_outline_0.csv", index=False, sep=";")
#%% Complete mapping
cats_inital = pd.read_csv(resources_path + "ddc_wikicat_mapping.csv", sep=";")
cats_manual_ext = pd.read_csv(resources_path + "ddc_wikicat_mapping_manual_extension.csv", sep=";")
cats_outline_ext = pd.read_csv(resources_path + "ddc_wikicat_mapping_outline_extension.csv", sep=";")
cats_complete = (pd.concat([cats_inital, cats_manual_ext, cats_outline_ext])) \
                    .sort_values(by="topic_id") \
                    .drop_duplicates() \
                    .reset_index(drop=True)
cats_complete.to_csv(resources_path + "ddc_wikicat_mapping_complete_0.csv", index=False, sep=";")

#%% Complete mapping with depth extension 1 to 5
cats_complete = pd.read_csv(resources_path + "ddc_wikicat_mapping_complete_0.csv", sep=";")
distinct_topic_ids = list(dict.fromkeys(cats_complete.topic_id.values))
all_depths = []
for i in [1,2,3,4,5]:
    cats_complete_depth_ext = []
    j = 0
    for topic_id in distinct_topic_ids:
        if j % 10 == 0: print(f"depth: {i} category: {j}")
        j += 1
        topic_df = cats_complete[(cats_complete.topic_id == topic_id)]
        cat_uris = list(topic_df.uri.values)
        res = DotMap(extend_wikicats(cat_uris, str(i)))
        for binding in res.results.bindings:
            uri = binding.subject.value
            name = ":".join(uri.split(":")[2:])
            cats_complete_depth_ext.append(WikiCat(topic_id=topic_id, name=name, uri=uri))
            
    cats_complete_depth_ext = pd.concat([pd.DataFrame(data=cats_complete_depth_ext), cats_complete])\
                                .sort_values(by="topic_id") \
                                .drop_duplicates() \
                                .reset_index(drop=True)
    cats_complete_depth_ext.to_csv(resources_path + f"ddc_wikicat_mapping_complete_{i}.csv", index=False, sep=";")
    all_depths.append(cats_complete_depth_ext)  
#%% enrich depth 2 with depth 1, depth 3 with depth 2 ...
for i in [0,1,2,3,4]:
    first = pd.read_csv(resources_path + f"ddc_wikicat_mapping_complete_{i}.csv",  sep=";")
    second = pd.read_csv(resources_path + f"ddc_wikicat_mapping_complete_{i+1}.csv",  sep=";")
    new = pd.concat([first, second]) \
            .sort_values(by="topic_id") \
            .drop_duplicates() \
            .reset_index(drop=True)
    new.to_csv(resources_path + f"ddc_wikicat_mapping_complete_{i+1}.csv", sep=";", index=False)
    
#%% Collect pages from all used categories for corpus construction 1
# batchsize := number of categories for which abstracts are requested
# batches := how many requests need to be made to cover all categories
categories_d5 = pd.read_csv(resources_path + "ddc_wikicat_mapping_complete_5.csv", sep=";")
distinct_categories_d5 = list(dict.fromkeys(categories_d5.uri.values))
distinct_categories_d5.sort()
batchsize = 50
batches = len(distinct_categories_d5) // batchsize if (len(distinct_categories_d5) % batchsize == 0) else len(distinct_categories_d5) // batchsize + 1
#%% collect pages from all used categories for corpus construction 2
# batchchunk_size := how many batches are compiled together to save per file
# start_batch := at which batch to start
# batch_save_it := iterator for naming files - increments every batchchunk_size 
batch_save_it = 0
start_batch = 0
batchchunk_size = 50
pages = []
for i in [*range(batches)]:
    if i < start_batch: continue
    lower_bound = i * batchsize
    upper_bound = min((i+1) * batchsize, len(distinct_categories_d5))
    cats = distinct_categories_d5[lower_bound:upper_bound]
    res = get_wiki_pages_and_abstracts_for_wikicats(cats)
    while res == -1:
        get_wiki_pages_and_abstracts_for_wikicats(cats)
    res = DotMap(res)
    for binding in res.results.bindings:
        cat = binding.o.value
        page = binding.sub.value
        abstract = binding.abstract.value
        pages.append(WikiPage(cat_uri=cat, page_uri=page, abstract=abstract))
    print(f"batch: {i+1} \t cats: {upper_bound} of {len(distinct_categories_d5)} \t pages: {len(pages)}")
    if (i+1) % batchchunk_size == 0:
        pages_df = pd.DataFrame(data=pages)
        pages_df.to_csv(resources_path + f"page_data/batchchunk_{batch_save_it}.csv", index=False, sep=";")
        batch_save_it +=1
        pages = []
#%% compile all pages to one df 
path_page_data = resources_path + "page_data/"
filename_page_data = "batchchunk_$.csv"
batchchunk_max = 416
batchchunk_min = 0
dfs = []
for i in [*range(batchchunk_max+1)]:
    print(f"load chunk {i}")
    try:
        file = path_page_data + filename_page_data.replace("$", str(i))
        dfs.append(pd.read_csv(file, sep=";"))
    except:
        print("empty file")
print("sorting")
df = pd.concat(dfs) \
    .sort_values(by="cat_uri") \
    .drop_duplicates() \
    .reset_index(drop=True)   
df.to_csv(path_page_data + "batchchunk_complete.csv", index=False, sep=";")

#%% Split complete_page_data in 20 even chunks, as 10gb is too much data to work with at once
file = resources_path + "page_data/batchchunk_complete.csv"
df = pd.read_csv(file, sep=";")
chunks = 20
chunk_size = len(df.index) // 20
for i in [*range(chunks)]:
    lower_bound = i * chunk_size
    upper_bound = (i+1) * chunk_size
    if i == chunks - 1:
        upper_bound = len(df.index)
    df[lower_bound : upper_bound] .to_csv(resources_path + f"page_data/data_sorted_{i}.csv", index=False, sep=";")

#%% Constructing corpus with pages and wikicat data
# Decoding:
# initial parse _ manual extension _ outline extension _ depth extension
# Go through every sorted page_data and try to find pages belonging to a given corpus mapping
# Save data every 400000 pages found
pd.options.mode.chained_assignment = None  # default='warn'
def get_pages_class_mapping(ddc_cat_df, cat_page_df):
    result = []
    topic_id = ""
    page_uri = ""
    page_abstract = ""
    cat_lower_bound = cat_page_df.loc[cat_page_df.index[0], "cat_uri"]
    cat_upper_bound = cat_page_df.loc[cat_page_df.index[-1], "cat_uri"]
    skip = 10000
    skips = len(cat_page_df.index) // skip
    cat_page_df_cat_index = []
    
    for i in [*range(skips + 1)]:
        lower_bound = i * skip
        upper_bound = min((i+1) * skip, len(cat_page_df.index))
        cat_page_df_cat_index.append({
            "index" : [lower_bound, 
                       upper_bound],
            "cats" : [cat_page_df.loc[cat_page_df.index[lower_bound], "cat_uri"], 
                      cat_page_df.loc[cat_page_df.index[upper_bound-1], "cat_uri"]]
            })
        
    
    for row in ddc_cat_df.itertuples():
        cat_uri = row.uri
        topic_id = row.topic_id
        if cat_uri >= cat_lower_bound and cat_uri <= cat_upper_bound:
            for i in cat_page_df_cat_index:
                if cat_uri >= i["cats"][0] and cat_uri <= i["cats"][1]:
                    df = cat_page_df[i["index"][0]:i["index"][1]].query("cat_uri == @cat_uri")
                    df.reset_index(drop=True)
                    df["topic_id"] = [topic_id for i in [*range(len(df.index))]]
                    df = df[["topic_id", "page_uri", "abstract"]]
                    result += df[["topic_id", "page_uri", "abstract"]].values.tolist()
    return result 

page_data_files = [resources_path + f"page_data/data_sorted_{i}.csv" for i in [*range(20)]]  
mapping_files = {
    "1_0_0_0" : resources_path + "ddc_wikicat_mapping.csv",
    "0_1_0_0" : resources_path + "ddc_wikicat_mapping_manual_extension.csv",
    "0_0_1_0" : resources_path + "ddc_wikicat_mapping_outline_extension.csv",
    "1_1_1_0" : resources_path + "ddc_wikicat_mapping_complete_0.csv",
    "1_1_1_1" : resources_path + "ddc_wikicat_mapping_complete_1.csv",
    "1_1_1_2" : resources_path + "ddc_wikicat_mapping_complete_2.csv",
    "1_1_1_3" : resources_path + "ddc_wikicat_mapping_complete_3.csv",
    "1_1_1_4" : resources_path + "ddc_wikicat_mapping_complete_4.csv",
    "1_1_1_5" : resources_path + "ddc_wikicat_mapping_complete_5.csv"
    }
chunk_threshold = 400000
save_to = resources_path + "corpus/base/"

for key, value in mapping_files.items():
    ddc_page_data = []
    t_chunk = []
    t_pages_collected = []
    ddc_cat_df = pd.read_csv(value, sep=";")
    chunk_it = 0
    print(key)
    t = datetime.now()
    for i in page_data_files:        
        cat_page_df = pd.read_csv(i, sep=";")
        # print(f"chunk loaded      {datetime.now()-t}")
        t_chunk.append(datetime.now()-t)
        t = datetime.now()
        ddc_page_data += get_pages_class_mapping(ddc_cat_df, cat_page_df)
        print(f"{i.split('/')[-1]} \t   {datetime.now()-t}")
        t_pages_collected.append(datetime.now()-t)
        while len(ddc_page_data) > chunk_threshold:
            temp_df = pd.DataFrame(data=ddc_page_data[:chunk_threshold], columns= ["topic_id", "page_uri", "abstract"])
            temp_df.to_csv(save_to + key + "-chunk" + str(chunk_it) + ".csv", index=False, sep=";")
            chunk_it += 1
            del temp_df
            ddc_page_data = ddc_page_data[chunk_threshold:]
        t = datetime.now()    
        del cat_page_df
        gc.collect()
    temp_df = pd.DataFrame(data=ddc_page_data, columns= ["topic_id", "page_uri", "abstract"])
    temp_df.to_csv(save_to + key + "-chunk" + str(chunk_it) + ".csv", index=False, sep=";")
    del temp_df 
    del ddc_page_data
    gc.collect()
    print(sum(t_pages_collected[1:], t_pages_collected[0]))
#%% Group found corpus data by ddc class


grp_by_class_path = resources_path + "corpus/grp_by_ddc/"
files = [[i.split('\\')[-1].split("-")[0], i] for i in glob.glob(resources_path + "corpus/base/*")]
ddc_top_100 = [*range(0,1000,10)]
it = 0
for file in files:
    t = datetime.now()
    corpus = file[0]
    print(f"{it} {corpus}")
    df = pd.read_csv(file[1], sep=";")
    for i in ddc_top_100:
        topic_data = df.query(f"topic_id >= {i} and topic_id < {i+10}")
        ## save to grp_by_class
        if len(topic_data.index) == 0:
            continue
        path = grp_by_class_path + corpus 
        if not os.path.exists(grp_by_class_path + corpus):
                os.makedirs(grp_by_class_path + corpus)
        file_path = path + f"/{i}_{it}.csv"
        topic_data.to_csv(file_path, index=False, sep=";")
    print(datetime.now()-t)
    it+=1
#%% combine them  
grp_by_class_path = resources_path + "corpus/grp_by_ddc/" 
folders = glob.glob(grp_by_class_path + "*")
ddc_top_100 = [*range(0,1000,10)]
for folder in folders:
    for i in ddc_top_100:
        files = glob.glob(f"{folder}/{i}_*.csv")
        if not files:
            continue
        dfs = [pd.read_csv(file, sep=";") for file in files]
        df = pd.concat(dfs)
        df.to_csv(folder + f"/{i}.csv", index=False, sep=";")
        # remove old files
        for file in files:
            os.remove(file)
            
    


            
