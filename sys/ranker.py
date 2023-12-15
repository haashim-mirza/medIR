"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization,
and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
from tqdm import tqdm


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A list containing tuples of the documents (ids) and their relevance scores
        """
        query_parts = self.tokenize(query)
        if self.stopwords != None and len(self.stopwords) > 0:
            query_parts = [None if token in self.stopwords else token for token in query_parts]
        query_word_counts = Counter(query_parts)
        query_word_counts.pop(None, None)

        docids = []
        for token in query_word_counts.keys():
            if token in self.index.index.keys():
                docids += [tup[0] for tup in self.index.index[token]]
        docids = set(docids)

        doc_term_counts = {}
        for doc in tqdm(docids, 'ranker.query: basic document term frequencies gather run'):
            doc_term_counts[doc] = {}
            for token in query_word_counts.keys():
                for post in self.index.get_postings(token):
                    if post[0] == doc:
                        doc_term_counts[doc][token] = post[1]

        results = []
        for doc in tqdm(docids, 'ranker.query: result scoring run'):
            results.append((doc, self.scorer.score(doc, doc_term_counts[doc], query_word_counts)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """

    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)
        """
        raise NotImplementedError

class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        doc_counts = []
        query_counts = []
        for term in set(doc_word_counts.keys()).union(set(query_word_counts.keys())):
            doc_val = doc_word_counts.get(term)
            query_val = query_word_counts.get(term)
            if doc_val == None:
                doc_counts.append(0)
            else:
                doc_counts.append(doc_val)
            if query_val == None:
                query_counts.append(0)
            else:
                query_counts.append(query_val)
        
        return sum([x * y for x, y in zip(doc_counts,query_counts)])

class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        stats = self.index.get_statistics()
        doc_metadata = self.index.get_doc_metadata(docid)
        N = stats['number_of_documents']

        scores = []
        for term in doc_word_counts.keys():
            df_t = len(self.index.index[term])
            qtf = ((self.k3 + 1) * query_word_counts[term]) / (self.k3 + query_word_counts[term])
            dtf = ((self.k1 + 1) * doc_word_counts[term]) / (self.k1 * (1 - self.b + (self.b * doc_metadata['doc_length'] / stats['mean_document_length'])) + doc_word_counts[term])
            idf = np.log((N - df_t + 0.5) / (df_t + 0.5))
            scores.append(qtf * dtf * idf)
        return sum(scores)

class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        print(self.index.index)
        stats = self.index.get_statistics()
        print(stats)
        print('checking 0', stats['mean_document_length'])
        doc_metadata = self.index.get_doc_metadata(docid)

        scores = []
        for term in doc_word_counts.keys():
            qtf = query_word_counts[term]
            dtf = (1 + np.log(1 + np.log(doc_word_counts[term]))) / (1 - self.b + (self.b * doc_metadata['doc_length'] / stats['mean_document_length']))
            idf = np.log((stats['number_of_documents'] + 1) / len(self.index.index[term]))
            scores.append(qtf * dtf * idf)
        return sum(scores)

class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        stats = self.index.get_statistics()
        N = stats['number_of_documents']

        terms = list(set(query_word_counts.keys()).intersection(set(doc_word_counts.keys())))
        scores = []
        for term in terms:
            tf = np.log(doc_word_counts[term] + 1)
            idf = 1 + np.log(N / len(self.index.index[term]))
            scores.append(tf * idf)
        return sum(scores)
    
class TF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: "dict[str, int]", query_word_counts: "dict[str, int]") -> float:
        terms = list(set(query_word_counts.keys()).intersection(set(doc_word_counts.keys())))
        scores = []
        for term in terms:
            tf = np.log(doc_word_counts[term] + 1)
            scores.append(tf)
        return sum(scores)

class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        self.model = CrossEncoder(cross_encoder_model_name, max_length=512)
        self.text = raw_text_dict

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        if docid not in self.text.keys() or query == "":
            return 0

        return self.model.predict((query, self.text[docid]))
