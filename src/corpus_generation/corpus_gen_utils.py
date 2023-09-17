# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:43:06 2023

@author: Adrian Kuhn
"""

#%%
from dataclasses import dataclass, field, asdict
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import pickle



def save_obj(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
def load_obj(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def get_ddc_id_name_list (path_ddc_classes):
    f = open(path_ddc_classes)
    s = f.read()
    sSplit = s.split("\n")
    return [[i, sSplit[i]] for i in range(len(sSplit))]



def query_dbpedia(query_str, params=[], currenttry=0):
    sparql = SPARQLWrapper('https://dbpedia.org/sparql')
    sparql.setTimeout(120)
    query_replaced = query_str
    for param in params:
        query_replaced = query_replaced.replace(param[0], param[1])
    sparql.setReturnFormat(JSON)
    #print(query_replaced)
    sparql.setQuery(query_replaced)
    try:
        return sparql.query().convert()
    except:
        if currenttry < 100 :
            stime = min(60,(currenttry+1)*5)
            print(f"query failed sleep {stime}sec")
            time.sleep(stime)
            return query_dbpedia(query_str, params, currenttry+1)
        else:
            print("failed")
            return -1
    

@dataclass(order=True)
class WikiCat:
    topic_id: int = field()
    name: str = field(default="")
    uri: str = field(default="")
    
    def to_list(self):
        return [self.name, self.uri, self.topic_id]
    def get_csv_str(self):
        return str(self.topic_id) + ";" + self.name + ";" + self.uri
    
@dataclass()
class WikiPage:
    cat_uri: str = field(default="")
    page_uri: str = field(default="")
    abstract: str = field(default="")

