# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:30:51 2023

@author: Adrian Kuhn
"""

import databusclient
from glob import glob

url = "https://akswnc7.informatik.uni-leipzig.de/~mhofer/newsgroup20/20news-18828"  
path = "D:/Informatikstudium/bachelor/github/datasetclassification/20news-18828"
#%%
cats = glob(path + "/*")
datasets = []
for cat in cats:
    cat_name = cat.split("\\")[-1]
    files = glob(cat + "/*")
    files = [file.split("\\")[-1] for file in files]
    
    distributions = []
    print(cat)
    for file in files:
        distributions.append(
            databusclient.create_distribution(
                url=url + f"/{cat_name}/{file}", 
                cvs={"name":f"{file}"[:-4]},
                file_format="txt"))
        
    
    datasets.append(databusclient.create_dataset(
        version_id=f"https://databus.dbpedia.org/adrian1703/20news/{cat_name}/18828", 
        title=f"20newsgroups category subset: {cat_name}", 
        abstract= "20 Newsgroups dataset - a text classification corpus", 
        description="The 20 Newsgroups data set is a collection of approximately 18,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge, it was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews paper, though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering. (http://qwone.com/~jason/20Newsgroups/)", 
        license_url="https://www.gnu.org/licenses/lgpl-3.0.html",
        distributions=distributions))
    
#%%
for dataset in datasets:
    databusclient.deploy(dataset, "accf5851-de4f-4cc6-bc11-a6fdebe84a91")
