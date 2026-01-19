from src.store_chunks import store_chunks

def store_chunks_if_empty(all_chunks, collection):
    if collection.count() > 0:
        print(f"Vector store already contains {collection.count()} embeddings. Skipping ingestion.")
        return

    print("Vector store empty. Storing chunks...")
    store_chunks(all_chunks, collection)
