import os 
from dotenv import load_dotenv
from smolagents import DuckDuckGoSearchTool, LiteLLMModel, CodeAgent
from smolagents import tool
from tavily import TavilyClient
import requests

load_dotenv()
print("1. Environment loaded successfully!")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
print("2. Web Search Tool is ready!")

@tool
def web_search(query: str) -> str:
    """Performs a deep, factual web search using the Tavily Research API.
    
    Args:
        query: The search query to look up.
    Returns:
        Detailed factual summaries of the top search results.
    """
    try:
        # We use 'advanced' depth to force Tavily to read full articles
        response = tavily_client.search(query, search_depth="advanced")
        return str(response.get("results", "No results found."))
    except Exception as e:
        return f"Search failed: {str(e)}"
print("2. Tavily Web Search Tool is ready!")


@tool
def read_text_file(file_path:str) -> str:
    """Reads the content of a text file.

    Args:
        file_path: The path of the file to read.
    Returns:
        The content of the file as a string.
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

print("3. File Reader Tool is ready!")

import pypdf
@tool
def read_local_pdf(file_path: str) -> str:
    """Reads the content of a local PDF document.
    
    Args:
        file_path: The absolute path of the PDF file to read.
    Returns:
        The extracted text content of the PDF.
    """
    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF {file_path}: {str(e)}"

print("X. PDF Reader Tool is ready!")


@tool
def search_person(query: str) -> str:
    """Uses the Happenstance API to fetch precise background and professional intelligence about a person.
    
    Args:
        query: The name, company, or social handle of the person you are researching.
    Returns:
        A summarized professional profile of the target individual.
    """
    try:
        api_key = os.getenv("HAPPENSTANCE_API_KEY")
        if not api_key:
            return "Error: HAPPENSTANCE_API_KEY is not configured in the environment."
            
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # We hit the search endpoint with our networking query
        response = requests.post(
            "https://api.happenstance.ai/v1/search",
            headers=headers,
            json={"text": query, "include_my_connections": True},
            timeout=10
        )
        response.raise_for_status()
        
        search_id = response.json().get("id")
        
        # 2. Wait and Poll for intelligence data
        import time
        for _ in range(6):  # Poll every 5 seconds, max 30s
            time.sleep(5)
            poll_resp = requests.get(f"https://api.happenstance.ai/v1/search/{search_id}", headers=headers, timeout=10)
            data = poll_resp.json()
            
            if data.get("status") == "COMPLETED":
                results = data.get("results")
                if results and len(results) > 0:
                    return "Here is the Happenstance Data: " + str(results)
                else:
                    return "Happenstance Search completed successfully, but no matching profile data was found."
                    
        return f"Happenstance API is taking too long to compile the profile. The search is running at: {response.json().get('url')}"
        
    except Exception as e:
        return f"People Search failed: {str(e)}"

print("4. Happenstance Search Tool is ready!")


# 3. Initialize the Core Engine (The Groq Brain)
model_id = "groq/llama-3.3-70b-versatile"

print(f"Loading '{model_id}' engine...")
model = LiteLLMModel(
    model_id=model_id,
    api_key=os.getenv("GROQ_API_KEY") 
)
print("4. Brain Engine is online!")


# 5. Define the System Prompt
SYSTEM_PROMPT = """You are Kaal Agent, an elite, highly capable AI assistant designed to assist and provide information to users, use tools, and write local Python code to solve complex queries.

Identity rule: If a user asks "who are you", always respond EXACTLY with: "I'm Kaal Agent, designed to assist and provide information to users."

Follow these rules strictly:
1. Break the problem into small, logical sub-tasks. Think step-by-step.
2. Use the `web_search` and `read_text_file` tools to gather facts.
3. Use the built-in python interpreter to execute logic and calculations.
4. ERROR HANDLING: If an execution fails or throws a traceback, do not give up. Read the error, fix your Python code logically, and run it again.
5. FINAL OUTPUT: When you have the exact solution, return it as concisely as possible (usually a single number, name, or date). Do NOT include conversational filler in your final answer unless asked.
"""

print("5. System prompt defined!")

# 6. Instantiate the CodeAgent
print("Constructing the Kaal CodeAgent...")
agent = CodeAgent(
    tools=[web_search, read_text_file, read_local_pdf, search_person], # The Senses include People Search and PDF Reader!
    model=model, # The Brain
    add_base_tools=True, # This tells smolagents to include the built-in Python Interpreter tool!
    additional_authorized_imports=["requests", "re", "json", "time", "datetime", "math", "os"]
)
# Modify the predefined system prompt of smolagents CodeAgent to inject our instructions
agent.system_prompt = SYSTEM_PROMPT + "\n\n" + agent.system_prompt
print("6. Kaal Agent is fully assembled and ready for complex tasks! 🚀")


# Agent engine successfully assembled and ready for import!
