# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 00:14:36 2023

@author: Adrian Kuhn
"""

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import stem_text
from gensim.parsing.preprocessing import strip_short

def tokenize_gensim(doc): 
    tokens = preprocess_string(doc, [
        lambda x: x.lower(), 
        lambda x: x.replace('\'', ''), 
        strip_tags, 
        strip_punctuation, 
        strip_numeric, 
        remove_stopwords, 
        lambda x: strip_short(x, minsize=3), 
        stem_text])
    return tokens

def count_tokens(token_list):
    token_dict = {}
    for token in token_list:
        token_dict.setdefault(token, 0)
        token_dict[token] += 1
    return token_dict


def tokenize_df(ddc_class, df, target_dir):
    abstract_word_count_col = [len(abstract.split(" ")) for abstract in df["abstract"]]
    abstract_character_count_col = [len(abstract) for abstract in df["abstract"]]
    token_col = [count_tokens(tokenize_gensim(abstract)) for abstract in df["abstract"]]
    df["abstract_word_count"]       = abstract_word_count_col
    df["abstract_character_count"]  = abstract_character_count_col
    df["tokens"]                    = token_col
    df = df[["topic_id", "page_uri", "abstract_word_count", "abstract_character_count", "tokens"]]
    df.to_csv(f"{target_dir}/{ddc_class}.csv", index=False, sep=";")