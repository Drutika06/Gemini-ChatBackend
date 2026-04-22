import os
import sys
import datetime
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# TOOL 1: Date and Time
# ==========================================
def get_current_date_and_day() -> str:
    """ALWAYS use this tool to get the current exact date and day of the week whenever the user asks."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y")

# ==========================================
# TOOL 2: Weather API (OpenWeatherMap)
# ==========================================
def get_weather(location: str) -> str:
    """Returns the current weather and temperature for a given city or location."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or api_key == "your_openweather_api_key_here":
        return "System Error: OPENWEATHER_API_KEY not configured."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
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
    if not api_key or api_key == "your_news_api_key_here":
        return "System Error: NEWS_API_KEY not configured."

    url = "https://newsdata.io/api/1/latest"
    params = {"apikey": api_key, "q": topic, "language": "en"}
    try:
        response = requests.get(url, params=params)
        data = response.json()
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
# CLI CHAT LOOP
# ==========================================
def main():
    print("Initializing Gemini AI...")
    chat_session = initialize_chat()
    print("Ready! Type your message (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            response = chat_session.send_message(user_input)
            if response.parts:
                text_responses = [part.text for part in response.parts if hasattr(part, 'text') and part.text]
                if text_responses:
                    print(f"\nAI: {' '.join(text_responses)}\n")
                else:
                    print("\nAI: [Tool executed successfully, but no summary was generated.]\n")
            else:
                print("\nAI: [Empty response received.]\n")
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
