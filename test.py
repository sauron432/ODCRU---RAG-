from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model = 'llama3.2')

template = """
You are an expert in answering question about airlines reviews.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

result = chain.invoke({"reviews":["I complained and asked for the alternative, vermicelli noodles with some stir fried chicken. It wasn't great, too much vermicelli and not enough sauce, but it was adequate. An hour or so later, a senior hostess came and chatted with me, and asked me what was wrong, I described the egg, she said someone else had also complained, I thanked her for asking me. I had an extremely good Nasi Lemak going up to Bangkok from Singapore on January 8th with Scoot, so the Changi kitchens can produce good."], "question": "Complaint about food."})

print(result)