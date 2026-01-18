from src.config import *
from src.chunker import create_chunks
from src.embedding import get_embeddings
from src.search import search
from src.preprocess_query import preprocess_query

print("------ Initialising the RAG pipeline ------")

def main():
    all_chunks = create_chunks(DATA_DIR, COLUMN_NAME)

    # print(all_chunks) 
    embeddings = get_embeddings(all_chunks)
    # print(embeddings)
    user_query = input("\nUser: ")
    user_query = preprocess_query(user_query)
    output = search(user_query, embeddings, all_chunks)
    print(output)
    
if __name__ == "__main__":
    main()