import os
from dotenv import load_dotenv
from smolagents import InferenceClientModel, CodeAgent, tool
from tavily import TavilyClient
import datasets
import requests
import io
from pypdf import PdfReader

load_dotenv()

# --- 1. SETUP SENSES ---
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def read_pdf_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts all its text. Use this when you find a .pdf link that holds the answer.
    Args:
        url: The direct link to the .pdf file.
    Returns:
        The extracted text from the entire PDF.
    """
    try:
        # Download the raw binary data of the PDF
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Load the binary data into the PDF Reader
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        # Loop through all the pages and glue the text together
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # If it's too long, we warn the LLM
        if len(text) > 30000:
            return text[:30000] + "\n...[TRUNCATED FOR LENGTH]"
            
        return text if text else "Error: PDF appears to be empty or unreadable."
    except Exception as e:
        return f"Failed to read PDF: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Performs a deep, factual web search using the Tavily Research API.
    Args:
        query: The search query to look up.
    Returns:
        Detailed factual summaries of the top search results.
    """
    try:
        response = tavily_client.search(query, search_depth="advanced")
        return str(response.get("results", "No results found."))
    except Exception as e:
        return f"Search failed: {str(e)}"

# --- 2. SETUP BRAIN ---
model = InferenceClientModel(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    token=os.getenv("HF_TOKEN")
)

agent = CodeAgent(
    tools=[web_search, read_pdf_url], # Add the new tool here!
    model=model,
    add_base_tools=True
)


print("Agent is ready for benchmarking!")


# --- 3. THE TRUE GAIA RUN ---
print("\nDownloading GAIA Benchmark Dataset...")

# We load GAIA's validation split (which contains the answer keys)
try:
    gaia_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=os.getenv("HF_TOKEN"))
    print(f"Dataset downloaded! Total questions available: {len(gaia_ds)}")
except Exception as e:
    print(f"Could not download dataset: {e}")
    exit()

# Counters for our loop
score = 0
total_tested = 0
MAX_TESTS = 3 # Let's test just 3 questions so it doesn't take an hour!

print(f"\n🚀 Start Testing (Target: {MAX_TESTS} Questions)")

for row in gaia_ds:
    # 1. Skip questions that require attached files for this simple test
    if row.get("file_name", "") != "":
        continue
        
    question = row["Question"]
    real_answer = row["Final answer"]
    
    total_tested += 1
    
    print(f"\n==============================================")
    print(f"[Question {total_tested}]: {question}")
    
    # 2. Add strict formatting rules to the prompt
    prompt = f"""
    You are an elite reasoning AI taking a strict benchmark test.
    QUESTION: {question}
    
    CRITICAL RULES:
    1. Use the web_search tool to find accurate information.
    2. Your final output must literally be NOTHING MORE than the exact answer value.
    """
    
    # 3. Ask the Agent to solve it (This starts the thoughts > actions loop)
    try:
        agent_answer = agent.run(prompt)
        
        print("\n--- GRADING ---")
        print(f"Agent's Answer: {agent_answer}")
        print(f"Real Answer:    {real_answer}")
        
        # 4. Simple grading: Did the agent find the exact fact?
        if str(real_answer).strip().lower() in str(agent_answer).strip().lower():
            print("✅ CORRECT!")
            score += 1
        else:
            print("❌ INCORRECT!")
    except Exception as e:
        print(f"❌ FAILED (Agent crashed during reasoning): {e}")

    # Stop after we test 3 questions without file attachments
    if total_tested == MAX_TESTS:
        break

print(f"\n==============================================")
print(f"🏆 Final Score: {score} out of {total_tested} Correct!")
