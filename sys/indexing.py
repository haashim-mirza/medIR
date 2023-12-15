from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from document_preprocessor import Tokenizer
import gzip
import bisect

class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        self.statistics = defaultdict(Counter)
        self.index = {}
        self.document_metadata = {}

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
              A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save().

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError

class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        super().__init__()
        self.vocabulary = set()
        self.statistics['unique_token_count'] = 0
        self.statistics['total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['mean_document_length'] = 0

    def remove_doc(self, docid: int) -> None:
        total_tokens_removed = 0
        keys = list(self.index.keys())
        for i in range(len(keys)):
            key = keys[i]
            val = self.index[key]
            ind = bisect.bisect_left(val, docid, key=lambda x: x[0])
            if val[ind] == docid:
                total_tokens_removed += val.pop(ind)[1]
                
                if len[val] == 0:
                    del self.index[key]

        self.vocabulary = set(self.index.keys())
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] -= total_tokens_removed
        self.statistics['number_of_documents'] -= 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents'] if self.statistics['number_of_documents'] > 0 else 0
    
    def add_doc(self, docid: int, tokens: "list[str]") -> None:
        # add to vocabulary
        # self.vocabulary = self.vocabulary.union(set(tokens))
        # if None in self.vocabulary:
        #     self.vocabulary.remove(None)

        # update stats
        self.statistics['total_token_count'] += len(tokens)
        self.statistics['number_of_documents'] += 1
        # self.statistics['unique_token_count'] = len(self.vocabulary)
        # self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
        self.document_metadata[docid] = [len(tokens), len(set(tokens)) - 1] if None in set(tokens) else [len(tokens), len(set(tokens))]

        token_counts = Counter(tokens)
        token_counts.pop(None, None)
        for token, count in token_counts.items():
            tup = (docid, count)
            if token in self.index.keys():
                self.index[token].append(tup)
            else:
                self.index[token] = [tup]

    def get_postings(self, term: str) -> "list[tuple[int, int]]":
        if term not in self.index.keys():
            return []
        return self.index[term]
    
    def get_doc_metadata(self, doc_id: int) -> "dict[str, int]":
        doc_metadata = self.document_metadata[doc_id]
        return {'doc_length': doc_metadata[0], 'num_unique_terms': doc_metadata[1]}
    
    def get_term_metadata(self, term: str) -> "dict[str, int]":
        return {'occurrences': sum([x[1] for x in self.index[term]]), 'num_docs': len(self.index[term])}
    
    def get_statistics(self) -> "dict[str, int]":
        return self.statistics
    
    def save(self, index_directory_name) -> None:
        dir = os.path.join(os.getcwd(), index_directory_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, "index.json")

        print("serializing index")

        serial_index = {
            "index": self.index,
            "stats": self.statistics,
            "vocab": list(self.vocabulary),
            "doc_metadata": self.document_metadata
        }

        print("dumping json")

        with open(path, 'w') as f:
            json.dump(serial_index, f, indent=2)
    
    def load(self, index_directory_name) -> None:
        path = os.path.join(index_directory_name, "index.json")
        with open(path, 'r') as f:
            serial_index = json.load(f)
            self.index = serial_index["index"]
            self.statistics = serial_index["stats"]
            self.vocabulary = set(serial_index["vocab"])
            self.document_metadata = dict(zip([int(k) for k in list(serial_index["doc_metadata"].keys())], list(serial_index["doc_metadata"].values())))

class Indexer:
    """
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    """
    @staticmethod
    
    def create_index(dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str], text_key="text", 
                     doc_augment_key: str="", foldername: str="") -> InvertedIndex:
        """
        Creates an inverted index.

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text
                and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index
        """
        
        index = BasicInvertedIndex()
        word_freqs = {}
        with open(dataset_path) as f:
            num_docs = sum(1 for _ in tqdm(f, "finding max_docs"))
        print("found num_docs")

        with open(dataset_path, 'r') as f:
            for _ in tqdm(range(num_docs), "counting corpus frequencies"):
                doc = json.loads(f.readline())
                docid = doc["docid"]
                text = doc[text_key]
                if doc_augment_key != "":
                    for q in doc[doc_augment_key]:
                        text += " " + q

                tokens = document_preprocessor.tokenize(text)
                for tok in tokens:
                    if tok in word_freqs.keys():
                        word_freqs[tok] += 1
                    else:
                        word_freqs[tok] = 1
        print("got corpus word frequencies")
        
        with open(dataset_path, 'r') as f:
            for _ in tqdm(range(num_docs), "adding docs"):
                doc = json.loads(f.readline())
                docid = doc["docid"]
                text = doc[text_key]
                if doc_augment_key != "":
                    for q in doc[doc_augment_key]:
                        text += " " + q

                tokens = document_preprocessor.tokenize(text)

                if stopwords != None and len(stopwords) > 0:
                    tokens = [None if token in stopwords else token for token in tokens]
                
                index.add_doc(docid, tokens)

        print("all documents added")

        for _, docfreqs in tqdm(index.index.items(), "sorting index values"):
            docfreqs.sort(key=lambda x: x[0])

        print("sorting complete")

        index.vocabulary = set(index.index.keys())
        index.statistics['unique_token_count'] = len(index.vocabulary)
        index.statistics['mean_document_length'] = index.statistics['total_token_count'] / index.statistics['number_of_documents']
        index.statistics['body'] = "1" if text_key == "text" else "0"

        print("statistics updated")

        if foldername == "":
            index.save(text_key + "_index")
        else:
            index.save(foldername)

        print("save complete")

        return index

