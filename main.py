import os
import sys
import datetime
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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



# 3. Initialize the AI brain 
chat_session = initialize_chat()

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
        response = chat_session.send_message(request.message)
        
        if response.parts:
            text_responses = [part.text for part in response.parts if hasattr(part, 'text') and part.text]
            if text_responses:
                final_reply = " ".join(text_responses)
            else:
                final_reply = "[Tool executed successfully, but no summary was generated.]"
        else:
            final_reply = "[Empty response received.]"
            
        return ChatResponse(reply=final_reply)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))