import pandas as pd
import os
import re

def clean_text(text):
    """
    Function to clean the text by removing special characters, digits, converting to lowercase,
    and removing extra whitespace. Removes rows containing 'page not found' in the text column.
    """
    # Check if 'page not found' is present in the text
    if 'page not found' in text.lower():
        return None  # Return None to signal removal of the entire row
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'10|&#10;|&|#|â€™', '', cleaned_text)
    cleaned_text = re.sub(r'faq.*', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'read more.*', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'also read.*', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'read more:*', '', cleaned_text)
    cleaned_text = re.sub(r'alisha.*', '', cleaned_text)
    cleaned_text = re.sub(r'sara.*', '', cleaned_text)
    return cleaned_text.strip()

def preprocess_text(text, chunk_size=500):
    """
    Function to preprocess the text by cleaning it and breaking it into chunks of specified size.
    """
    # Clean the text
    text = clean_text(text)
    if text is None:
        return []  # Return an empty list if the text is None (row should be skipped)
    # Split the text into words
    words = text.split()
    # Break down the text into chunks of chunk_size words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def filter_unwanted_urls(url):
    """
    Function to filter unwanted URLs based on keywords.
    """
    unwanted_keywords = ['infographic', 'category', 'tag', 'archive']
    return not any(keyword in url for keyword in unwanted_keywords)

def process_csv(input_file_path, chunk_size=500):
    """
    Function to process the CSV file, apply text preprocessing, filter unwanted URLs,
    and save the processed data to a new CSV file.
    """
    # Load the CSV file
    data = pd.read_csv(input_file_path)

    # Drop rows where 'page not found' is in the text column
    data = data[~data['text'].str.lower().str.contains('page not found')]

    # Filter unwanted URLs
    data = data[data['url'].apply(filter_unwanted_urls)]

    # Apply the preprocessing function to the text_chunk column
    data['cleaned_text'] = data.apply(lambda row: preprocess_text(row['text'], chunk_size), axis=1)

    # Create a new DataFrame with each chunk of text, title, and URL
    processed_data = []
    for index, row in data.iterrows():
        for chunk in row['cleaned_text']:
            processed_data.append([chunk, row['title'], row['url']])
    
    # Convert the list to a DataFrame
    processed_df = pd.DataFrame(processed_data, columns=['text_chunk', 'title', 'url'])

    # Add a new 'id' column filled with numbers from 1 to n
    processed_df['id'] = range(1, len(processed_df) + 1)

    # Determine the output file path
    output_file_path = os.path.join(os.path.dirname(input_file_path), 'preprocessed_data_cleaned.csv')

    # Save the processed DataFrame to the output CSV file
    processed_df.to_csv(output_file_path, index=False)

    print(f"Processed data saved to {output_file_path}")

input_file_path = 'cleaned_data2.csv'  # Replace with your input file path
process_csv(input_file_path, chunk_size=500)
