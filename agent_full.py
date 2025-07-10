# RAG Model: retrieval-augmented generation

from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from ics import Calendar, Event
from dotenv import load_dotenv
import os
import tempfile
import json

load_dotenv()


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

trip_calendar = Calendar()

def add_to_calendar(event_json: str) -> str:
    data = json.loads(event_json)
    e = Event()
    e.name, e.begin, e.end = data["title"], data["start"], data["end"]
    trip_calendar.events.add(e)

    path = os.path.join(tempfile.gettempdir(), "trip_events.ics")
    with open(path, "w") as f:
        f.writelines(trip_calendar)
    return f"Calendar written to {path}"

def find_kid_friendly_sfo_activities(_: str) -> str:
    """Return 3–4 concrete kid-friendly activities in San Francisco."""
    activities = [
        {"title": "Exploratorium at Pier 15",
         "start": "2025-07-25T10:00", "end": "2025-07-25T13:00"},
        {"title": "Aquarium of the Bay",
         "start": "2025-07-25T14:00", "end": "2025-07-25T16:00"},
        {"title": "Cable-Car Ride & Hyde St Pier",
         "start": "2025-07-26T10:00", "end": "2025-07-26T12:00"},
        {"title": "California Academy of Sciences",
         "start": "2025-07-26T13:30", "end": "2025-07-26T16:30"},
    ]
    return json.dumps(activities)

tools = [
    Tool(name="BookFlight", func=mock_book_flight, description="Books a flight given details, returns JSON"),
Tool(name="FindActivities", func=find_kid_friendly_sfo_activities, description="Returns JSON array of kid-friendly activities for San Francisco trip"),
Tool(
    name="AddToCalendar",
    func=add_to_calendar,
    description=(
        "Use this to add a new event to the calendar. "
        "Input must be a JSON string with the keys: "
        "`title` (event title), `start` (ISO datetime), and `end` (ISO datetime). "
    )
)
]

# Initialize the actual Agent with Memory + Tools

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature = 0)

prefix = (
    "You are a travel-planning assistant.\n"
    "For **every** flight, hotel night, and daytime activity you create,\n"
    "call the `AddToCalendar` tool separately, so that the final calendar\n"
    "contains one VEVENT per item. Do not stop until all items are added."
)

agent = initialize_agent(
    tools, llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
)

# Run Agent

user_request = """

Plan and book a family trip from NYC to San Francisco for July 24-27.

Include: 
- Outbound flight
- A hotel near Fisherman's Wharf
- A minimum of three kid-friendly activities during the day
- Add all itinerary items into the calendar
"""
result = agent.invoke({"input": user_request})
print(result)