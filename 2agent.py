# ----------------- imports & setup -----------------
from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from ics import Calendar, Event
import os
import json
import tempfile
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectordb = FAISS.from_texts(
    ["Family prefers late morning flights", "Kids need vegetarian meals"], embeddings
)
memory = VectorStoreRetrieverMemory(retriever=vectordb.as_retriever(search_kwargs={"k": 3}))

# ------------- global calendar that accumulates -------------
trip_calendar = Calendar()
calendar_path = os.path.join(tempfile.gettempdir(), "trip_events.ics")

def add_to_calendar(event_json: str) -> str:
    """Accept either one event (dict) or a list of events, add them all."""
    events = json.loads(event_json)
    if isinstance(events, dict):   # single event
        events = [events]

    for data in events:
        e = Event()
        e.name  = data["title"]
        e.begin = data["start"]
        e.end   = data["end"]
        trip_calendar.events.add(e)

    with open(calendar_path, "w") as f:
        f.writelines(trip_calendar)

    return f"{len(events)} event(s) written → {calendar_path}"

# ------------------- mock tools -------------------
def mock_book_flight(details: str) -> str:
    return json.dumps({"flight_id": "XY123", "details": details,
                       "title": "Flight – NYC to San Francisco",
                       "start": "2025-07-24T08:00", "end": "2025-07-24T11:00"})

def mock_book_hotel(_: str) -> str:
    return json.dumps({"hotel_name": "Holiday Inn Express Fisherman's Wharf",
                       "title": "Hotel – Holiday Inn Express Fisherman's Wharf",
                       "start": "2025-07-24", "end": "2025-07-27"})

def find_kid_friendly_sfo_activities(_: str) -> str:
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
    Tool("BookFlight", mock_book_flight, "Books a flight, returns JSON with title/start/end"),
    Tool("BookHotel",  mock_book_hotel,  "Books the hotel, returns JSON with title/start/end"),
    Tool("FindActivities", find_kid_friendly_sfo_activities,
         "Returns JSON array of kid-friendly activities for SF"),
    Tool("AddToCalendar", add_to_calendar,
         "Add one event to the calendar; input is JSON with title,start,end")
]

# --------------- LLM + agent with prefix ---------------
prefix = (
    "You are a travel-planning assistant.\n"
    "Required workflow:\n"
    "1. Book the flight with BookFlight.\n"
    "2. Book the hotel with BookHotel.\n"
    "3. Find activities with FindActivities.\n"
    "4. For **each** item (flight, hotel stay, every activity) call AddToCalendar "
    "separately before you finish."
)

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
agent = initialize_agent(
    tools, llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
    agent_kwargs={"prefix": prefix}   # ← this is how the LLM sees the instruction
)

# ----------------- run -----------------
user_request = """
Plan and book a family trip from NYC to San Francisco for July 24–27.
Include:
- Outbound flight
- A hotel near Fisherman's Wharf
- A minimum of three kid-friendly daytime activities
- Add all itinerary items into the calendar
"""
agent.invoke({"input": user_request})

print("\nOpen", calendar_path, "to see the full itinerary.")