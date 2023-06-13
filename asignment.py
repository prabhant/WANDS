import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, TFAutoModel
from sentence_transformers import SentenceTransformer

#defining class for FAISS Library
class FaissIdx:
    def __init__(self, model, dim=384):
        self.index = faiss.IndexFlatIP(dim)
        # Maintaining the document data
        self.doc_map = dict()
        self.model = model
        self.ctr = 0

    def add_doc(self, document_text: pd.Series, doc_id: pd.Series) -> None:
        self.index.add(np.reshape(self.model.encode(document_text),(1,384)))
        self.doc_map[self.ctr] = (document_text,doc_id) # store the original document text
        self.ctr += 1

    def search_doc(self, query: str, k=1) -> list:
        D, I = self.index.search(np.reshape(self.model.encode(query),(1,384)), k)
        return [{self.doc_map[idx]: score} for idx, score in zip(I[0], D[0]) if idx in self.doc_map]

#Read the data
products_df = pd.read_csv('dataset/product.csv' , delimiter='\t')
labels_df = pd.read_csv('dataset/label.csv', delimiter='\t')
queries_df = pd.read_csv('dataset/query.csv', delimiter='\t')
#Add a score in the labels dataframe when label is 'Exact' score is 1 when label is 'Partial' score is 0.5 when label is 'Irrelevant' score is 0
labels_df['score'] = labels_df['label'].apply(lambda x: 1 if x == 'Exact' else 0.5 if x == 'Partial' else 0)
#cleaning up pandas dataset with relevant columns

product_emb_df = products_df[['product_id', 'product_name', 'product_description']]
#removing NAN values
product_emb_df = product_emb_df.dropna()

#Defining the model
model = SentenceTransformer('all-MiniLM-L12-v2')
#encoding the product description and product name
encodings_st = model.encode(list(product_emb_df['product_description']))
#Defining the FAISS index
index = FaissIdx(model)
for i in range(len(list(product_emb_df['product_description']))):
  prod_id, prod_doc = product_emb_df['product_id'].iloc[i], product_emb_df['product_description'].iloc[i]
  index.add_doc(prod_doc, prod_id)

#scoring the model based on the label score and the product id returned by the FAISS index for the query
total_score = 0
for i in range(len(list(queries_df['query']))):
  query = queries_df['query'].iloc[i]
  results = index.search_doc(query, k=1)
  for result in results:
    for key, value in result.items():
      print(key, value)
      print(labels_df[labels_df['product_id'] == key[1]]['score'].iloc[0])
      score = labels_df[labels_df['product_id'] == key[1]]['score'].iloc[0]
      total_score += score
      print('\n')
#calculating the score
print(total_score)