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

# Define Tools for the AI agent to use

def mock_book_flight(details: str) -> str:
    # TODO: call an airline API/flight tracker API
    return json.dumps({"flight_id": "XY123", "details": details})

def add_to_calendar(event_json: str) -> str:
    event = Event()
    data = json.loads(event_json)
    event.name = data["title"]
    event.begin = data["start"]
    event.end = data["end"]
    calendar = Calendar()
    calendar.events.add(event)
    path = tempfile.gettempdir() + "/trip_events.ics"
    with open(path, "w") as f: f.writelines(calendar)
    return f"Calendar written to {path}"

