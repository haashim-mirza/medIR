from numpy import ndarray
from ranker import Ranker
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class VectorDBRanker(Ranker):
    def __init__(self, raw_id_to_text: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = FAISS.load_local("symptomdb.faiss", self.embeddings)
        self.maps = raw_id_to_text

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.
        Performs query expansion using pseudo-relevance feedback if needed.

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: We don't use the user_id parameter in vector ranker. It is here just to align all the
                    Ranker interfaces.

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        scored_docs = self.db.similarity_search_with_relevance_scores(query, top_k=3)
        tups = []
        for key, score in scored_docs:
            for docid, text in self.maps.items():
                if key.page_content == text:
                    tups.append((docid, score))
        return tups


# import json

# docid_to_text = {}
# with open("final_data.json", "r") as f:
#     for line in f:
#         raw_data = json.loads(line)
#         docid_to_text[int(raw_data["docid"])] = raw_data["text"]


# instance = VectorDBRanker(raw_id_to_text=docid_to_text)
# print(instance.query("I have a headache"))
