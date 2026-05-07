from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import os

tavily_search = TavilySearch(max_results=3, search_engine="google", api_key= os.getenv("TAVILY_API_KEY"))

class Tools:
    tools = [tavily_search]
