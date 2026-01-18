import pandas as pd
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import *

def create_chunks(df_path, text_column):
    print("------ Creating chunks ------")
    all_chunks = []
    chunks = []
    df = pd.read_csv(df_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = OVERLAP_SIZE)
    # reviews = df[text_column]
    # print(reviews)
    for _, row in df.iterrows():
        review = row[COLUMN_NAME]
        airline = row[AIRLINE]
        chunks.extend(text_splitter.split_text(review))
        for chunk in chunks:
            all_chunks.append({
                "text":chunk,
                "airline":airline
            })
    # print(f"Total chunks: {len(all_chunks)}")
    all_chunks = all_chunks[:50]
    return all_chunks
