# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 18:37:58 2023

@author: Adrian Kuhn
"""

#%% parse data from https://en.wikipedia.org/wiki/List_of_Dewey_Decimal_classes
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import requests
from bs4 import BeautifulSoup
from corpus_generation.corpus_gen_utils import WikiCat
import pandas as pd
#%%
def parse_dewey_wiki_page():
    url = "https://en.wikipedia.org/wiki/List_of_Dewey_Decimal_classes"
    r = requests.get(url, allow_redirects=True)
    soup = BeautifulSoup(r.content, 'html.parser')
    a = soup.find_all("li")
    links = soup.find_all('a')
    b_tags = list(soup.findAll('b'))
    b = [str(x) for x in a]
    b = b[13:1090] 
    c = [x for x in b if "<ul>" in x]
    d = []
    for cat in c:
        link_text = []
        for link in links:
            s = str(link.get("href"))
            if s in cat and s.startswith('/wiki'):
                link_text.append(link.get("href"))
        cat_name = ""
        for b_tag in b_tags:
            if str(b_tag.contents[0]) in cat:
                cat_name = str(b_tag.contents)
                cat_name = cat_name[1:].strip()[1:5].strip()
                
                break
        d.append([cat_name, link_text])
    for entry in d:
        link_list = []
        for link in entry[1]:
            link_list += parse_outline_pages(link)
        entry[1] = list((dict.fromkeys(link_list)).keys())
    return d
def parse_outline_pages(outline_link):
    url = "https://en.wikipedia.org" + outline_link
    if not url.startswith("https://en.wikipedia.org/wiki/Outline"): return [outline_link]
    r = requests.get(url, allow_redirects=True)
    soup = BeautifulSoup(r.content, 'html.parser')
    links = soup.find_all('a')
    link_text = []
    for link in links:
        s = str(link.get("href"))
        if not s.startswith("/wiki"): continue
        if s.startswith("/wiki/Help:"): continue 
        if s.startswith("/wiki/Wikipedia:"): continue 
        if s.startswith("/wiki/Special:"): continue 
        if s.startswith("/wiki/File:"): continue 
        if s.startswith("/wiki/Portal:"): continue
        link_text.append(link.get("href"))
    link_text = list((dict.fromkeys(link_text)).keys())
    return link_text

def get_wikicat_from_dewey_outline():
    dewey_link_list = parse_dewey_wiki_page()
    cats = []
    for entry in dewey_link_list:
        pages = entry[1]
        categories = [link for link in entry[1] if link.startswith("/wiki/Category:")]
        categories = ['http://dbpedia.org/resource/' + link[6:] for link in categories]
        filter_strings = ["outlines",
                          "Outlines",
                          "Harv"
                          "Wikipedia",
                          "Wikidata",
                          "Pages",
                          "pages",
                          "Articles",
                          "articles",
                          "Annotated_link",
                          "Use_dmy_dates_from_April_2017",
                          "Wikipedia_missing_topics"]
        for category in categories:
            cat_name = category.split(":")[-1]
            skip = False
            for filt in filter_strings:
                if filt in cat_name: skip = True
            if skip or cat_name.startswith("_") or cat_name.endswith("_errors"): continue
            cats.append(WikiCat(topic_id=int(entry[0]), name=cat_name, uri=category ))
    return pd.DataFrame(data=cats)