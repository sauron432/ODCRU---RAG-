import ollama
import numpy as np
from typing import List

def search(query, collection, threshold = 0.8):
    # default_message = "The query provided is out of context of my knowledge base. I don't have enough information to answer this."    
    try:
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=query)["embedding"]
        # print("------ Generating Response ------") 
        
    except Exception as e:
        print(f"Errpr embedding query: {e}")
        
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results = 5
    )
    
    responses = [
        {
            "text": results["documents"][0][i],
            "airline": results["metadatas"][0][i]["airline"]
            # "score": results["distances"][0][i]
        } 
        for i in range(len(results["documents"][0]))
    ]
    # best_response = min(responses, key=lambda x: x["score"])
    # if best_response["score"] > threshold:
    #     return default_message
    # return best_response['text']
    return responses
    # return threshold
    # return f"Review:{responses[0]['text']} \n Airline: {responses[0]['airline']} " 


