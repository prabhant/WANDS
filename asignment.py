import pandas as pd
#Read the data
products_df = pd.read_csv('dataset/product.csv' , delimiter='\t')
labels_df = pd.read_csv('dataset/label.csv', delimiter='\t')
queries_df = pd.read_csv('dataset/query.csv', delimiter='\t')
#Add a score in the labels dataframe when label is 'Exact' score is 1 when label is 'Partial' score is 0.5 when label is 'Irrelevant' score is 0
labels_df['score'] = labels_df['label'].apply(lambda x: 1 if x == 'Exact' else 0.5 if x == 'Partial' else 0)

product_emb_df = products_df[['product_id', 'product_name', 'product_description']]
product_emb_df = product_emb_df.dropna()
from transformers import AutoTokenizer, TFAutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L12-v2')

encodings_st = model.encode(list(product_emb_df['product_description']))

import numpy as np
import faiss
class FaissIdx:
    def __init__(self, model, dim=384):
        self.index = faiss.IndexFlatIP(dim)
        # Maintaining the document data
        self.doc_map = dict()
        self.model = model
        self.ctr = 0

    def add_doc(self, document_text):
        self.index.add(np.reshape(self.model.encode(document_text),(1,384)))
        self.doc_map[self.ctr] = document_text # store the original document text
        self.ctr += 1

    def search_doc(self, query, k=3):
        D, I = self.index.search(self.model.get_embedding(query), k)
        return [{self.doc_map[idx]: score} for idx, score in zip(I[0], D[0]) if idx in self.doc_map]

faiss_idx = FaissIdx(model)
for doc in list(product_emb_df['product_description']):
    faiss_idx.add_doc(doc)

