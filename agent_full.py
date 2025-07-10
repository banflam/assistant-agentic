from langchain.agents import initialize_agent, Tool
from langchain.llms import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from ics import Calendar, Event
import os
import tempfile
import json

# Memory setup using Vector DB as the vector database

# Embeddings are vectors of text to capture their semantic meaning
# Initialize a wrapper around OpenAI's embedding model
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Store prior user preferences and instructions it in a FAISS database for search purposes
vectordb = FAISS.from_texts(["Family prefers late morning flights",
                             "Kids need vegetarian meals"], embeddings)

# Create a retriever object to handle embedding the query (transform into a similar vector)
memory = VectorStoreRetrieverMemory(
    retriever=vectordb.as_retriever(search_kwargs={"k": 3})
)