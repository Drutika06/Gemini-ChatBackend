import os
import sys
import datetime
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone

# ==========================================
# TOOL 1: Date and Time
# ==========================================
def get_current_date_and_day() -> str:
    """ALWAYS use this tool to get the current exact date and day of the week whenever the user asks."""
    print("DEBUG: get_current_date_and_day() called")
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y")

# ==========================================
# TOOL 2: Weather API (OpenWeatherMap)
# ==========================================
def get_weather(location: str) -> str:
    """Returns the current weather and temperature for a given city or location."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    print(f"DEBUG: get_weather() called with location={location!r}")
    
    if not api_key or api_key == "your_openweather_api_key_here":
        return "System Error: OPENWEATHER_API_KEY not configured."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        print(f"DEBUG: OpenWeatherMap response status: {response.status_code}")
        
        if response.status_code == 200:
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            return f"The weather in {location} is currently {temp}°C with {description}."
        else:
            return f"Could not fetch weather for {location}. API said: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"System error fetching weather: {str(e)}"

# ==========================================
# TOOL 3: News API (NewsData.io)
# ==========================================
def get_latest_news(topic: str) -> str:
    """ALWAYS use this tool to fetch the top 3 latest news headlines. 
    CRITICAL: The 'topic' must be a single simple keyword like 'India' or 'Technology'."""
    api_key = os.getenv("NEWS_API_KEY")
    print(f"DEBUG: get_latest_news() called with topic={topic!r}")
    
    if not api_key or api_key == "your_news_api_key_here":
        return "System Error: NEWS_API_KEY not configured."

    url = "https://newsdata.io/api/1/latest"
    params = {"apikey": api_key, "q": topic, "language": "en"}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        print(f"DEBUG: NewsData.io response status: {response.status_code}") 
        
        if response.status_code == 200 and data.get("status") == "success":
            articles = data.get("results", [])
            if not articles:
                return f"No news found for the topic: {topic}."
            
            headlines = [f"- {art.get('title', 'No Title')} (Source: {art.get('source_id', 'Unknown')})" for art in articles[:3]]
            return "\n".join(headlines)
        else:
            return f"News API Error: {data.get('results', 'Unknown error')}"
    except Exception as e:
        return f"System error fetching news: {str(e)}"

# ==========================================
# AI INITIALIZATION
# ==========================================
def initialize_chat():
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("CRITICAL ERROR: GEMINI_API_KEY is missing.")
        sys.exit(1)
        
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(
            model_name='gemini-3.1-flash-lite-preview', 
            tools=[get_current_date_and_day, get_weather, get_latest_news],
            system_instruction="You are a helpful AI backend agent. Use tools to fetch live data when needed."
        )
        return model.start_chat(history=[], enable_automatic_function_calling=True)
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        sys.exit(1)


def init_pinecone_index():
    """Connect to an existing Pinecone index created by setup_pinecone.py."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "mbsr-rag-index")

    if not pinecone_api_key:
        print("CRITICAL ERROR: PINECONE_API_KEY is missing.")
        sys.exit(1)

    try:
        pc = Pinecone(api_key=pinecone_api_key)
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            print(f"CRITICAL ERROR: Pinecone index '{index_name}' not found.")
            print("Run: python3 setup_pinecone.py --create")
            sys.exit(1)
        return pc.Index(index_name)
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        sys.exit(1)


def estimate_query_complexity(query: str) -> int:
    """Choose adaptive top_k for retrieval based on query complexity."""
    word_count = len(query.split())
    question_count = query.count("?")
    complexity = word_count + (question_count * 2)

    if complexity < 4:
        return 2
    if complexity < 10:
        return 4
    return 5


def retrieve_from_pinecone(query: str, top_k: int | None = None) -> list[dict]:
    """Retrieve relevant chunks from Pinecone using integrated query embedding."""
    namespace = os.getenv("PINECONE_NAMESPACE", "mbsr")
    if top_k is None:
        top_k = estimate_query_complexity(query)

    results = pinecone_index.search(
        namespace=namespace,
        query={"inputs": {"text": query}, "top_k": top_k},
    )

    hits = []
    for match in results.result.hits:
        fields = match.fields
        hits.append(
            {
                "score": match._score,
                "chunk_text": fields.get("chunk_text", ""),
                "source": fields.get("source", "unknown"),
                "page": fields.get("page", "?"),
            }
        )
    return hits


def rewrite_query(user_query: str, history: list[dict]) -> str:
    """Rewrite follow-up queries into standalone form for better retrieval."""
    if not history:
        return user_query

    history_str = "\n".join(
        f"{'User' if turn['role'] == 'user' else 'AI'}: {turn['content']}"
        for turn in history[-6:]
    )

    prompt = f"""Given the conversation history and the latest user query,
rewrite the query to be standalone and specific. Replace pronouns and references
with their actual subjects. If the query is already clear, return it unchanged.

CONVERSATION HISTORY:
{history_str}

LATEST QUERY: {user_query}

REWRITTEN QUERY (just the query, nothing else):"""

    response = rag_model.generate_content(prompt)
    if response.text:
        return response.text.strip().strip('"')
    return user_query


def ask_with_rag(question: str, history: list[dict]) -> str:
    """RAG flow: rewrite query, retrieve from Pinecone, then answer with Gemini."""
    rewritten = rewrite_query(question, history)
    hits = retrieve_from_pinecone(rewritten)

    if not hits:
        return "I could not find relevant context in the Pinecone knowledge base for that question."

    context_str = "\n\n".join(
        f"[Score: {hit['score']:.3f} | {hit['source']} p.{hit['page']}]\n{hit['chunk_text']}"
        for hit in hits
    )

    history_str = ""
    if history:
        recent = history[-6:]
        history_str = "RECENT CONVERSATION:\n" + "\n".join(
            f"{'User' if turn['role'] == 'user' else 'AI'}: {turn['content']}"
            for turn in recent
        ) + "\n\n"

    prompt = f"""You are an MBSR (Mindfulness-Based Stress Reduction) expert
answering questions about the MBSR handbook and mindfulness practice.

IMPORTANT GUIDELINES:
1. Ground your answer in the retrieved context.
2. Cover both practical structure and deeper mindfulness meaning when available.
3. Be complete but concise.
4. If context is missing, clearly state uncertainty and do not hallucinate.

{history_str}RETRIEVED CONTEXT:
{context_str}

CURRENT QUESTION: {question}

ANSWER:"""

    response = rag_model.generate_content(prompt)
    return response.text or "I could not generate an answer right now. Please try again."

# ==========================================
# FASTAPI APPLICATION SETUP
# ==========================================
# 1. Initialize the web framework
app = FastAPI(title="Gemini AI Backend API")

# 2. Add CORS Middleware (Crucial for connecting a frontend UI later!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # Local development
        "http://localhost:8000",           # Local FastAPI dev
        "http://127.0.0.1:3000",           # Local development (IP)
        "https://gemini-chat-79771110644.us-central1.run.app",  # Your Cloud Run domain
        "https://gemini-chat-gold.vercel.app",  # Your primary Vercel frontend
        "https://gemini-chat-98m2obx2t-drutika06s-projects.vercel.app", # <--- ADDED THIS LINE!
    ],
    allow_credentials=True,
    allow_methods=["*"], # Simplified to allow all methods
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)



# 3. Initialize AI + Pinecone clients
chat_session = initialize_chat()
rag_model = genai.GenerativeModel(model_name='gemini-3.1-flash-lite-preview')
pinecone_index = init_pinecone_index()
chat_history: list[dict] = []

# 4. Define the Data Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# ==========================================
# API ENDPOINTS
# ==========================================

# Silences the browser's automatic request for a tab icon
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/")
def health_check():
    return {
        "status": "Online",
        "message": "Welcome to the Gemini API Backend. Send a POST request to /chat to talk to the AI."
    }

@app.options("/chat")
def options_chat():
    return {"message": "OK"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_gemini(request: ChatRequest):
    try:
        final_reply = ask_with_rag(request.message, chat_history)
        chat_history.append({"role": "user", "content": request.message})
        chat_history.append({"role": "assistant", "content": final_reply})
        return ChatResponse(reply=final_reply)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))