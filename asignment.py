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


