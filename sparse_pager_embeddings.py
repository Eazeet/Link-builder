import pinecone
from pinecone_text.sparse import BM25Encoder
import pandas as pd

bm25 = BM25Encoder()


def main():
    data=pd.read_csv('data_cleaned.csv')
    bm25.fit(data['cleaned_text'])
    sparse_embeds = bm25.encode_documents([data['cleaned_text']])
    data['keywords']=sparse_embeds

    data.to_csv('data_with_keywords.csv')



if __name__=='__main__':
    main()
