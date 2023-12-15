from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer, WordCountCosineSimilarity
import json

class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        self.doc_index = document_index
        self.tit_index = title_index
        self.doc_preproc = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feat_extractor = feature_extractor
        self.model = LambdaMART()
                   
    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        X = []
        y = []
        qgroups = []

        for query, docrels in tqdm(query_to_document_relevance_scores.items(), "preparing features:query cycle"):
            qgroups.append(len(docrels))
            for docid, rel in tqdm(docrels, "preparing features:doc cycle"):
                query_parts = self.doc_preproc.tokenize(query)
                if self.stopwords != None and len(self.stopwords) > 0:
                    query_parts = [None if token in self.stopwords else token for token in query_parts]

                doc_word_counts = self.accumulate_doc_term_counts(index=self.doc_index, query_parts=query_parts)
                tit_word_counts = self.accumulate_doc_term_counts(index=self.tit_index, query_parts=query_parts)

                dwc = {}
                twc = {}
                if docid in doc_word_counts.keys():
                    dwc = doc_word_counts[docid]
                if docid in tit_word_counts.keys():
                    twc = tit_word_counts[docid]
                feats = self.feat_extractor.generate_features(docid=docid, doc_word_counts=dwc, title_word_counts=twc, query_parts=query_parts, query=query)

                X.append(feats)
                y.append(rel)

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        accumulated = {}
        for q_tok in set(query_parts):
            if q_tok is None:
                continue
            posts = index.get_postings(q_tok)
            for docid, freq in posts:
                if docid not in accumulated.keys():
                    accumulated[docid] = {}
                accumulated[docid][q_tok] = freq
        return accumulated

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        rel_ds = {}
        with open(training_data_filename, 'r') as f:
            train_data = json.load(f)
            rel_ds = {k:v['scored_docs'] for k, v in train_data.items()}
            
            # lines = f.readlines()
            # print(lines[0])
            # for line in tqdm(lines[1:], "preparing training data dict"):
            #     vals = line.strip('\n').split(',')
            #     k = vals[0]
            #     tup = (int(vals[1]), int(vals[2]))
            #     if k not in rel_ds.keys():
            #         rel_ds[k] = [tup]
            #     else:
            #         rel_ds[k].append(tup)
            #     line = f.readline()

        print("relevance dict created")

        X, y, qgroups = self.prepare_training_data(rel_ds)

        self.model.fit(X_train=X, y_train=y, qgroups_train=qgroups)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(featurized_docs=X)
    
    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: the integer id of the user who is issuing the query or None if the user is unknown
            mmr_lambda: Hyperparameter for MMR diversification scoring
            mmr_threshold: Documents to rerank using MMR diversification

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        if query is None or query == "":
            return []
        
        query_parts = self.doc_preproc.tokenize(query)
        if self.stopwords != None and len(self.stopwords) > 0:
            query_parts = [None if token in self.stopwords else token for token in query_parts]

        doc_term_counts = self.accumulate_doc_term_counts(index=self.doc_index, query_parts=query_parts)
        tit_term_counts = self.accumulate_doc_term_counts(index=self.tit_index, query_parts=query_parts)
        rel_docs = list(doc_term_counts.keys())
        if len(rel_docs) == 0:
            return []

        doc_scores = self.ranker.query(query)

        top_doc_scores = []
        bottom_doc_scores = []
        if (len(doc_scores) > 100):
            top_doc_scores = doc_scores[:100]
            bottom_doc_scores = doc_scores[100:]
        else:
            top_doc_scores = list(doc_scores)

        X = []
        for docid, _ in tqdm(top_doc_scores, 'l2r.query: extracting features for top docs'):
            dwc = {}
            twc = {}
            if docid in doc_term_counts.keys():
                dwc = doc_term_counts[docid]
            if docid in tit_term_counts.keys():
                twc = tit_term_counts[docid]
            X.append(self.feat_extractor.generate_features(docid=docid, doc_word_counts=dwc, title_word_counts=twc, query_parts=query_parts, query=query))

        prediction = self.predict(X)
        print('l2r.query: prediction calculated')

        new_top_doc_scores = [(top_doc_scores[i][0], prediction[i]) for i in range(len(top_doc_scores))]
        new_top_doc_scores.sort(key=lambda x: x[1], reverse=True)

        new_doc_scores = new_top_doc_scores + bottom_doc_scores
        final_rankings = [{'docid': int(tup[0]), 'score': tup[1]} for tup in new_doc_scores]

        return final_rankings


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        self.doc_index = document_index
        self.tit_index = title_index
        self.doc_cat_info = doc_category_info
        self.doc_preproc = document_preprocessor
        self.stopwords = stopwords
        self.ce_scorer = ce_scorer
        self.tittfs = []

        self.rec_cats = list(recognized_categories)

        self.doc_tf = TF(self.doc_index)
        self.tit_tf = TF(self.tit_index)
        self.doc_tfidf = TF_IDF(self.doc_index)
        self.tit_tfidf = TF_IDF(self.tit_index)
        self.bm25 = BM25(self.doc_index)
        self.pivnorm = PivotedNormalization(self.doc_index)
        self.wccossim = WordCountCosineSimilarity(self.doc_index)

    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.doc_index.document_metadata[docid][0]

    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.tit_index.document_metadata[docid][0]

    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        if self.stopwords != None and len(self.stopwords) > 0:
            query_parts = [None if token in self.stopwords else token for token in query_parts]
        query_word_counts = Counter(query_parts)
        query_word_counts.pop(None, None)

        if index.statistics["body"] == "1":
            return self.doc_tf.score(docid, word_counts, query_word_counts)
        else:
            return self.tit_tf.score(docid, word_counts, query_word_counts)

    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        if self.stopwords != None and len(self.stopwords) > 0:
            query_parts = [None if token in self.stopwords else token for token in query_parts]
        query_word_counts = Counter(query_parts)
        query_word_counts.pop(None, None)

        if index.statistics["body"] == "1":
            return self.doc_tfidf.score(docid, word_counts, query_word_counts)
        else:
            return self.tit_tfidf.score(docid, word_counts, query_word_counts)

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        if self.stopwords != None and len(self.stopwords) > 0:
            query_parts = [None if token in self.stopwords else token for token in query_parts]
        query_word_counts = Counter(query_parts)
        query_word_counts.pop(None, None)

        return self.bm25.score(docid, doc_word_counts, query_word_counts)

    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        if self.stopwords != None and len(self.stopwords) > 0:
            query_parts = [None if token in self.stopwords else token for token in query_parts]
        query_word_counts = Counter(query_parts)
        query_word_counts.pop(None, None)

        return self.pivnorm.score(docid, doc_word_counts, query_word_counts)
    
    def get_cosine_sim_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        if self.stopwords != None and len(self.stopwords) > 0:
            query_parts = [None if token in self.stopwords else token for token in query_parts]
        query_word_counts = Counter(query_parts)
        query_word_counts.pop(None, None)

        return self.wccossim.score(docid, doc_word_counts, query_word_counts)

    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        return [1 if cat in self.doc_cat_info[docid] else 0 for cat in self.rec_cats]

    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """
        score = self.ce_scorer.score(docid=docid, query=query)
        if score > 0:
            print('big ce here')
        return score

    def get_query_doc_word_inclusion_ratio(self, docid: int, query_parts: "list[str]") -> float:
        included = []
        for tok in set(query_parts):
            if tok is None:
                continue
            docids = []
            if tok in self.doc_index.index.keys():
                docids = [tup[0] for tup in self.doc_index.index[tok]]
            indicator = 1 if docid in docids else 0
            included.append(indicator)
        if len(included) == 0:
            return 0
        return sum(included) / len(included)
    
    def get_num_query_words_in_doc(self, docid: int, query_parts: "list[str]") -> float:
        num_words = 0
        for tok in set(query_parts):
            if tok is None:
                continue
            for doc, _ in self.doc_index.get_postings(tok):
                if doc == docid:
                    num_words += 1
                    break
        return num_words
    
    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        feature_vector = []

        # Document Length
        feature_vector.append(self.get_article_length(docid=docid))

        # Title Length
        feature_vector.append(self.get_title_length(docid=docid))

        # Query Length
        feature_vector.append(len(query_parts))

        # TF (document)
        feature_vector.append(self.get_tf(index=self.doc_index, docid=docid, word_counts=doc_word_counts, query_parts=query_parts))

        # TF-IDF (document)
        feature_vector.append(self.get_tf_idf(index=self.doc_index, docid=docid, word_counts=doc_word_counts, query_parts=query_parts))

        # TF (title)
        tf = self.get_tf(index=self.tit_index, docid=docid, word_counts=title_word_counts, query_parts=query_parts)
        self.tittfs.append(tf)
        feature_vector.append(tf)

        # TF-IDF (title)
        feature_vector.append(self.get_tf_idf(index=self.tit_index, docid=docid, word_counts=title_word_counts, query_parts=query_parts))

        # BM25
        feature_vector.append(self.get_BM25_score(docid=docid, doc_word_counts=doc_word_counts, query_parts=query_parts))

        # Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid=docid, doc_word_counts=doc_word_counts, query_parts=query_parts))
        
        # Word Count Cosine Similarity
        feature_vector.append(self.get_cosine_sim_score(docid=docid, doc_word_counts=doc_word_counts, query_parts=query_parts))

        # Cross-Encoder Score
        feature_vector.append(self.get_cross_encoder_score(docid=docid, query=query))

        # Query Words in Doc Ratio
        feature_vector.append(self.get_query_doc_word_inclusion_ratio(docid=docid, query_parts=query_parts))
        
        # Query Words in Doc
        feature_vector.append(self.get_num_query_words_in_doc(docid=docid, query_parts=query_parts))

        # Document Categories
        cat_feats = self.get_document_categories(docid=docid)
        feature_vector += cat_feats

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        self.lgbmranker = lightgbm.LGBMRanker(objective=default_params["objective"], boosting_type=default_params["boosting_type"], n_estimators=default_params["n_estimators"], importance_type=default_params["importance_type"], num_leaves=default_params["num_leaves"], learning_rate=default_params["learning_rate"], max_depth=default_params["max_depth"], n_jobs=default_params["n_jobs"])
  
    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        self.lgbmranker.fit(X=X_train, y=y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        return self.lgbmranker.predict(X=featurized_docs)

