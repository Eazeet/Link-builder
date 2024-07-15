import pinecone
from pinecone_text.sparse import BM25Encoder
import pandas as pd

bm25 = BM25Encoder()

def main():
    data = pd.read_csv('preprocessed_cleaned_data.csv')
    
    text_chunks = data['text_chunk'].astype(str).tolist()
    
    # Fit the encoder
    bm25.fit(text_chunks)
    
    # Encode the documents
    sparse_embeds = bm25.encode_documents(text_chunks)
    
    # Add the keywords to the DataFrame
    data['keywords'] = sparse_embeds
    
    # Save the DataFrame to a new CSV file
    data.to_csv('data_with_keywords.csv', index=False)

if __name__ == '__main__':
    main()
