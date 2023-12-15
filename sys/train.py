import csv
from collections import Counter, defaultdict
from tqdm import tqdm
import json
import numpy as np
import gzip
from datetime import datetime
from sentence_transformers import SentenceTransformer
import os
import re
import joblib

# your modules are imported here
from indexing import Indexer, BasicInvertedIndex
from document_preprocessor import RegexTokenizer, Doc2QueryAugmenter
from ranker import Ranker, BM25, CrossEncoderScorer
from vector_ranker import VectorRanker
from l2r import L2RFeatureExtractor, L2RRanker

data_prefix = '../data/'
model_prefix = '../models/'

with open(data_prefix + 'rec_cats.json', 'r') as f:
    rec_cats = json.load(f)
    five_cats = rec_cats['short']
    all_cats = rec_cats['full']
    
print('cats loaded')
    
with open(data_prefix + 'doc_cat_info.json', 'r') as f:
    doc_cat_info = json.load(f)
    doc_cat_info = {int(k):v for k, v in doc_cat_info.items()}
    
print('doc cat info loaded')
    
doc_preproc = RegexTokenizer('\\w+')

print('preprocessor initialized')

stopwords = set()
with open(data_prefix + 'stopwords.txt', 'r', encoding='utf-8') as file:
    for stopword in file:
        stopwords.add(stopword.strip())

print('stopwords loaded')

doc_base_index = BasicInvertedIndex()
doc_base_index.load(data_prefix + 'doc_base_index')
doc_small_index = BasicInvertedIndex()
doc_small_index.load(data_prefix + 'doc_small_index')
doc_flan_index = BasicInvertedIndex()
doc_flan_index.load(data_prefix + 'doc_flan_index')
doc_index = BasicInvertedIndex()
doc_index.load(data_prefix + 'doc_index')
tit_index = BasicInvertedIndex()
tit_index.load(data_prefix + 'title_index')

print('indices loaded')

with open(data_prefix + 'raw_text.json', 'r') as f:
    raw_text_dict = json.load(f)
with open(data_prefix + 'base_raw_text.json', 'r') as f:
    base_raw_text_dict = json.load(f)
with open(data_prefix + 'small_raw_text.json', 'r') as f:
    small_raw_text_dict = json.load(f)
with open(data_prefix + 'flan_raw_text.json', 'r') as f:
    flan_raw_text_dict = json.load(f)
    
print('raw text loaded')
    
ce_scorer = CrossEncoderScorer(raw_text_dict)
bce_scorer = CrossEncoderScorer(base_raw_text_dict)
sce_scorer = CrossEncoderScorer(small_raw_text_dict)
fce_scorer = CrossEncoderScorer(flan_raw_text_dict)

print('ce scorers initialized')

nn_feat_extract = L2RFeatureExtractor(doc_index, tit_index, doc_cat_info, doc_preproc, stopword, set(), ce_scorer)
nf_feat_extract = L2RFeatureExtractor(doc_index, tit_index, doc_cat_info, doc_preproc, stopword, set(five_cats), ce_scorer)
na_feat_extract = L2RFeatureExtractor(doc_index, tit_index, doc_cat_info, doc_preproc, stopword, set(all_cats), ce_scorer)
bf_feat_extract = L2RFeatureExtractor(doc_base_index, tit_index, doc_cat_info, doc_preproc, stopword, set(five_cats), bce_scorer)
sf_feat_extract = L2RFeatureExtractor(doc_small_index, tit_index, doc_cat_info, doc_preproc, stopword, set(five_cats), sce_scorer)
ff_feat_extract = L2RFeatureExtractor(doc_flan_index, tit_index, doc_cat_info, doc_preproc, stopword, set(five_cats), fce_scorer)
ba_feat_extract = L2RFeatureExtractor(doc_base_index, tit_index, doc_cat_info, doc_preproc, stopword, set(all_cats), bce_scorer)

print('feature extractors initialized')

nn_ranker = L2RRanker(doc_index, tit_index, doc_preproc, stopword, None, nn_feat_extract)
nf_ranker = L2RRanker(doc_index, tit_index, doc_preproc, stopword, None, nf_feat_extract)
na_ranker = L2RRanker(doc_index, tit_index, doc_preproc, stopword, None, na_feat_extract)
bf_ranker = L2RRanker(doc_base_index, tit_index, doc_preproc, stopword, None, bf_feat_extract)
sf_ranker = L2RRanker(doc_small_index, tit_index, doc_preproc, stopword, None, sf_feat_extract)
ff_ranker = L2RRanker(doc_flan_index, tit_index, doc_preproc, stopword, None, ff_feat_extract)
ba_ranker = L2RRanker(doc_base_index, tit_index, doc_preproc, stopword, None, ba_feat_extract)

print('rankers initialized')

nn_ranker.train(data_prefix + 'train_data.json')
joblib.dump(nn_ranker.model.lgbmranker, model_prefix + 'nn_model.joblib')

nf_ranker.train(data_prefix + 'train_data.json')
joblib.dump(nf_ranker.model.lgbmranker, model_prefix + 'nf_model.joblib')

na_ranker.train(data_prefix + 'train_data.json')
joblib.dump(na_ranker.model.lgbmranker, model_prefix + 'na_model.joblib')

bf_ranker.train(data_prefix + 'train_data.json')
joblib.dump(bf_ranker.model.lgbmranker, model_prefix + 'bf_model.joblib')

sf_ranker.train(data_prefix + 'train_data.json')
joblib.dump(sf_ranker.model.lgbmranker, model_prefix + 'sf_model.joblib')

ff_ranker.train(data_prefix + 'train_data.json')
joblib.dump(ff_ranker.model.lgbmranker, model_prefix + 'ff_model.joblib')

ba_ranker.train(data_prefix + 'train_data.json')
joblib.dump(ba_ranker.model.lgbmranker, model_prefix + 'ba_model.joblib')

print('rankers trained and dumped')