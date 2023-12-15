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

# your modules are imported here
from indexing import Indexer, BasicInvertedIndex
from document_preprocessor import RegexTokenizer, Doc2QueryAugmenter
from ranker import Ranker, BM25, CrossEncoderScorer
from vector_ranker import VectorRanker
from l2r import L2RFeatureExtractor, L2RRanker

data_prefix = '../data/'

with open(data_prefix + 'rec_cats.json', 'r') as f:
    rec_cats = json.load(f)
    top5cats = rec_cats['short']
    cats = rec_cats['full']
with open(data_prefix + 'doc_cat_info.json', 'r') as f:
    doc_cat_info = json.load(f)
    doc_cat_info = {int(k):v for k, v in doc_cat_info.items()}
    
