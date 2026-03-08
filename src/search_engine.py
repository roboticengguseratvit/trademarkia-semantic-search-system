import numpy as np

class SearchEngine:

    def __init__(self, index, documents):
        self.index = index
        self.documents = documents

    def search(self, query_embedding, k=5):

        D, I = self.index.search(query_embedding.reshape(1, -1), k)

        results = [self.documents[i] for i in I[0]]

        return results