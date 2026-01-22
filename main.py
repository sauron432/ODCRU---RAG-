from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

from src.config import *
from src.chunker import create_chunks
from src.vectorDB import create_vectorDB
from src.search import search
from src.check_for_chunks import store_chunks_if_empty
from src.preprocess_query import preprocess_query

print("------ Initialising the RAG pipeline ------")    

def main():
    all_chunks = create_chunks(DATA_DIR, COLUMN_NAME)
    collection = create_vectorDB()
    store_chunks_if_empty(all_chunks,collection)
    memory = ConversationBufferMemory(return_messages=True)
    print("\nRAG chatbot ready. Type 'exit' to quit.")
    while True:
        user_query = input("\nUser:")
        if user_query.lower() in {"exit"}:
            break
        user_query = preprocess_query(user_query)
        memory.chat_memory.add_message(HumanMessage(content=user_query))
        history = memory.load_memory_variables({})['history']
        limit = 4
        latest_history = history[-limit:]
        
        conversation_context = "\n".join([f"{msg.type}: {msg.content}" for msg in latest_history])
        # print(conversation_context)
        # print()
        enhanced_query = f"{conversation_context}\n\nCurrent Query: {user_query}"
        # print(enhanced_query)
        output = search(enhanced_query, collection)
        memory.chat_memory.add_message(AIMessage(content=output))
        print("\nResponse:",output)
        # print(memory.load_memory_variables({}))
    
if __name__ == "__main__":
    main()