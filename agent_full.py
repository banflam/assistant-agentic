from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

itinerary_formatter = LLMChain(
    llm = llm,
    prompt = PromptTemplate(
        input_variables=["raw_plain"],
        template="""
        Format the following plan into a clean Markdown itinerary with headers for each day, bullet points for activities, and clear time slots:

        {raw_plain}
"""
    )
)