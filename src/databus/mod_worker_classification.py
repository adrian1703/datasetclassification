# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:45:32 2023

@author: Adrian Kuhn
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from flask import Flask

from datetime import datetime
import pickle
import requests
from flask import request

def create_svm_classifier_mod(model_path, th, top_k):

    app = Flask(__name__)
    svm_model, svm_scaler, stm_model = 0,0,0
    
    with open("D:/Informatikstudium/bachelor/stp/src/svm.model", 'rb') as pickle_file:
        svm_model, svm_scaler, stm_model = pickle.load(pickle_file)
    if svm_model == 0 or svm_scaler == 0 or stm_model == 0:
        print("Error while loading model")
        exit()
        
        
    @app.route("/<publisher>/<group>/<artifact>/<version>/<file>/activity")
    def classify(publisher, group, artifact, version, file):
        t0 = datetime.now()
        download_url = request.args.get("source")
        
        page = requests.get(download_url).content
        page = (str(page)).replace("\n", "")
        
        scores = stm_model.get_class_scores_doc(page)
        scores_scaled = svm_scaler.transform(scores.reshape(1, -1))
        res = svm_model.predict_proba(scores_scaled)
        cat_predicted = [sorted(list(zip(weight, svm_model.classes_)), reverse=True)[:top_k]for weight in res]
        cat_predicted = [i for i in cat_predicted[0] if i[0] >= th]
        t1 = datetime.now()
        result = []
        if len(cat_predicted) == 0:
            cat_predicted = '"unclassified"'
        else:
            cats = [i[1] for i in cat_predicted]
            result = ''
            for i in cats:
                result += f'"{str(i)}", '
            result = result[:-2]
        return """   
@prefix mod: <http://dataid.dbpedia.org/ns/mod#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .


<activity.ttl> a mod:ClassificationMod ;
    mod:version "1.0.0" ;
    prov:used <{dataid}> ;
    prov:startedAtTime "{time_start}"^^xsd:dateTime ;
    prov:endedAtTime "{time_end}"^^xsd:dateTime ;
    mod:statSummary {predicted} .    
        """.format(dataid=download_url, time_start=t0, time_end=t1, predicted=result)
    
    return app


