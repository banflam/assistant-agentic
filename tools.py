from langchain.tools import Tool
from duckduckgo_search import DDGS

def web_search_tool(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return "\n".join([result["title"] + ": " + result["body"] for result in results[:3]])
    
# Create LangChain tool object to be able to expose this function to the LLM-powered agent
search_tool = Tool(
    name="Search",
    func=web_search_tool,
    description="Search the web for real-time information"
)    