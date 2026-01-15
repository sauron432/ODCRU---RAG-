from src.config import *
from src.chunker import create_chunks
from src.embedding import get_embeddings
from src.search import search

print("------ Initialising the RAG pipeline ------")
def main():
    all_chunks = create_chunks(DATA_DIR, COLUMN_NAME)

    # print(all_chunks) 
    embeddings = get_embeddings(all_chunks)
    # print(embeddings)
    
    output = search("Complaints about the seat?", embeddings, all_chunks)
    print("RESPONSE: ", output)
    
if __name__ == "__main__":
    main()