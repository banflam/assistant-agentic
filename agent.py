from langchain.agents import initialize_agent
from langchain.llms import openai

llm = openai(temperature=0)

tools = [search_tool]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)