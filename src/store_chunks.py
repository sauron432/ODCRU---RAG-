from src.config import *
import ollama

def store_chunks(all_chunks, collection):
    for i, chunk in enumerate(all_chunks):
        try:
            embedding = ollama.embeddings(
                model=MODEL,
                prompt=chunk["text"]
            )["embedding"]

            collection.add(
                ids=[str(i)],
                documents=[chunk["text"]],
                metadatas=[{"airline": chunk["airline"]}],
                embeddings=[embedding]
            )
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")

    print("All chunks stored. Total vectors:", collection.count())
