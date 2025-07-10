from langchain.tools import Tool
from duckduckgo_search import DDGS

def web_search_tool(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return "\n".join([result["title"] + ": " + result["body"] for result in results[:3]])
    
    