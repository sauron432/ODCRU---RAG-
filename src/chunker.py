import pandas as pd
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import *

def create_chunks(df_path, text_column: str) -> List[str]:
    """Takes a DataFrane path and the column name with text. Returns a list of first 50 chunks.

    Args:
        df_path (str): path of the DataFrame
        text_column (str): Name of column with text

    Returns:
        List[str]: chunks
    """
    print("------ Creating chunks ------")
    chunk_size = CHUNK_SIZE
    chunk_overlap = OVERLAP_SIZE
    all_chunks = []
    df = pd.read_csv(df_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    reviews = df[text_column]
    # print(reviews)
    for review in reviews:
        all_chunks.extend(text_splitter.split_text(review))
    # print(f"Total chunks: {len(all_chunks)}")
    all_chunks = all_chunks[:50]
    return all_chunks
