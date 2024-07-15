import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

openai_api_key = os.getenv('openai_api_key')
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # the maximum for text-embedding-3-small is 8191

client = OpenAI(api_key='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')

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def converts_string_to_list(string):
    """Converts the string representation of the embeddings list to a list of floats"""
    if not isinstance(string, str):
        raise ValueError(f"Expected a string, but got {type(string)}")
    if not (string.startswith('[') and string.endswith(']')):
        raise ValueError(f"Expected a string in list format, but got {string}")
    return [float(x) for x in string[1:-1].split(',')]

def main():
    # Load data 
    df = pd.read_csv('preprocessed_cleaned_data.csv')

    # Initialize tqdm for pandas
    tqdm.pandas()

    # Get embeddings with progress bar
    df['embeddings'] = df['text_chunk'].progress_apply(get_embedding)

    # Save the embeddings to a new csv file
    df.to_csv('text_with_embeddings.csv', index=False)

    # Load the dataframe with embeddings
    df = pd.read_csv('text_with_embeddings.csv')

    # Convert the string representation of the lists to a list of floats
    df['embeddings'] = df['embeddings'].progress_apply(converts_string_to_list)

    # Save the final embeddings to a new csv file
    df.to_csv('final_embeddings_ada.csv', index=False)

if __name__ == "__main__":
    main()
