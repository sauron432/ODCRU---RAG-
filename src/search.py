import ollama
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


def search(query, embeddings, all_chunks):
    
    if embeddings is None:
        print("No stored embeddings found!")
    
    try:
        print("------ Generating Response ------") 
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=query)["embedding"]
        
    except Exception as e:
        print(f"Errpr embedding query: {e}")

    cos = cosine_similarity(
        [query_embedding],
        embeddings
        )[0]
    
    best_index = np.argmax(cos)
    return f" Answer:{all_chunks[best_index]['text']} \n Airline: {all_chunks[best_index]['airline']} \n Score: {all_chunks[best_index]['score']} " 


