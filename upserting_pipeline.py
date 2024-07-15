import os
from dotenv import load_dotenv
import time
from pinecone import Pinecone,ServerlessSpec
import concurrent.futures
import pandas as pd
import json
import numpy as np
import ast

load_dotenv()
# load the final embeddings
embeddings = pd.read_csv("final_embeddings_ada.csv")
sparse = pd.read_csv("data_with_keywords.csv")
embeddings['keywords'] = sparse['keywords']

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv('pinecone_api_key')

# configure client
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec('aws', 'us-east-1')

# define index name
index_name='hybridsearch2'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embeddings-small
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
print(index.describe_index_stats())



batch_size = 100

# Iterate over the DataFrame in batches
for start in range(0, len(embeddings), batch_size):
    end = min(start + batch_size, len(embeddings))
    batch_json_data = []

    # Construct JSON objects for the current batch
    for ind, row in embeddings.iloc[start:end].iterrows():
        sparse_values = ast.literal_eval(row['keywords'])
        json_obj = {
            "id": str(row['id']),
            "values": list(map(float, row['embeddings'].strip('[]').split(','))),
            "sparse_values":sparse_values,
            "metadata": {
                "text": row['text_chunk'],
                "url": row['url'],
                "title": row['title']
            }
        }
        batch_json_data.append(json_obj) 

    # Upsert the current batch
    index.upsert(batch_json_data,namespace='hybridsearch2')




print('done')

print(index.describe_index_stats())