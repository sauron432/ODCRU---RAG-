# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from src.config import *
from src.search import search
from src.chunker import create_chunks
from src.vectorDB import create_vectorDB
from src.check_for_chunks import store_chunks_if_empty
from src.preprocess_query import preprocess_query

print("------ Initialising the RAG pipeline ------")    

def main():
	 # Vector DB setup
    all_chunks = create_chunks(DATA_DIR, COLUMN_NAME)
    collection = create_vectorDB()
    store_chunks_if_empty(all_chunks, collection)
    # Prompt engineering 
    prompt = ChatPromptTemplate.from_messages([
        (
        "system","""You are an assistant that answers questions ONLY using airline reviews.
        If the answer is not explicitly supported by the provided reviews or chat history,respond with a brief refusal.
        Do not comment on whether a previous answer was correct or incorrect.
        Do not guess, paraphrase missing information, or explain past mistakes."""),
        MessagesPlaceholder("chat_history"),
        ("human","Here are the airline reviews:\n{reviews}\n\nUser question:\n{input}")
    ])
    # LLM initialization 
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature = TEMP
    )
    # For history storing 
    store = {}
    # RAG chain
    input_extractor = RunnableLambda(lambda x: x["input"])
    history_extractor = RunnableLambda(lambda x: x.get("chat_history", []))
    retriever = RunnableLambda(lambda q: search(q, collection))
    rag_chain = (
        {
            "reviews": input_extractor | retriever,
            "input": input_extractor,
            "chat_history": history_extractor,
        }
        | prompt | llm
    )
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: store.setdefault(session_id, ChatMessageHistory()),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    # Main chat loop
    print("\nConversational RAG chatbot ready. Type 'exit' to quit.")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == "exit":
            break
        user_query = preprocess_query(user_query)
        response = conversational_chain.invoke(
            {"input": user_query},
            config={"configurable": {"session_id": USER}}
        )
        print("\nAssistant:", response.content)

if __name__ == "__main__":
    main()