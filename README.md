# Conversational RAG Chatbot (LangChain + Ollama)

A Retrieval-Augmented Generation (RAG) chatbot that answers user questions from airline reviews and while maintaining conversation history. The project uses LangChain , a vector database(chromaDB), and Ollama for local LLM inference. The data is collected from kaggle in .csv format and extracted using pandas. The reviews are processed and stored into the vectorDB from where the agent retrieves similar reviews using cosine distance from the vectorDB. The LLM uses the reviews returned by the vectorDB along with the chat history and user query to generate responses.

---

##  Features

* **RAG pipeline** – retrieves relevant airline reviews before answering
* **Conversation memory** – chat history stored per session
* **Local LLM** – powered by Ollama (no cloud dependency)
* **Modular design** – clean separation of search, chunking, embeddings, and chains

---

## Project Structure

```
project-root/
│
├── notebook/
│   └── notebook.ipynb         # Notebook file for testing
├── src/
│   ├── __init__.py            # Recognizes src as package 
│   ├── search.py              # Vector search logic
│   ├── chunker.py             # Document chunking
│   ├── vectorDB.py            # Vector DB creation & persistence
│   ├── preprocess_query.py    # Query cleaning/preprocessing
│   ├── check_for_chunks.py    # Ensures chunks exist before search
│   ├── store_chunks.py        # Stores chunks in vectorDB 
│   └── config.py              # Global config & constants
├── data/
│   └── airlines_reviews.csv   # Original airline reviews
├── test.py                    # Testing the LLM 
├── main.py                    # Chat loop & RAG chain setup
├── requirements.txt
└── README.md
```

---

##  Architecture Overview

```
User Input
   ↓
RunnableWithMessageHistory
   ↓
Input Extractor
   ↓
Vector Search (Reviews)
   ↓
ChatPromptTemplate
   ↓
Ollama LLM
   ↓
Response
```

Conversation history is automatically appended and reused on every turn.

---

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running locally and the required model is pulled:

```bash
ollama pull llama3.2
```

---

## Prompt Design

The system prompt strictly enforces grounded answers:

* Answers must come only from airline reviews or chat history
* If not supported → the model must refuse briefly
* No guessing, no paraphrasing missing info

---

## Running the Chatbot

```bash
python main.py
```

Example interaction:

```
User: complaints about food
Assistant: Passengers have complained about watery meals and poorly prepared noodles...
```
---
