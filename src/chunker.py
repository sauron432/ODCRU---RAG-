import pandas as pd
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import *

# df_path = "../data/airlines_reviews.csv"
# text_column = "Reviews"
def create_chunks(df_path, text_column):
    print("------ Creating chunks ------")
    all_chunks = []
    chunks = []
    df = pd.read_csv(df_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100)
    # reviews = df[text_column]
    # print(reviews)
    for _, row in df.iterrows():
        review = row[text_column]
        airline = row['Airline']
        chunks = text_splitter.split_text(review)
        for chunk in chunks:
            all_chunks.append({
                "text":chunk,
                "airline":airline
            })
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks[:1875]
