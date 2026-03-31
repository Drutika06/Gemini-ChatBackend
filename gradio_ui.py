import gradio as gr
import requests

# The URL where our FastAPI backend server is listening
BACKEND_URL = "http://127.0.0.1:8000/chat"

def chat_with_backend(message, history):
    """
    This function takes the user's message, sends it to the FastAPI backend, 
    and returns the AI's response to be displayed in the Gradio UI.
    """
    try:
        # Send the POST request to your FastAPI backend
        response = requests.post(
            BACKEND_URL, 
            json={"message": message}
        )
        
        # Check if the backend responded successfully
        if response.status_code == 200:
            # Extract the "reply" string and return it to the UI
            bot_reply = response.json().get("reply", "No reply received.")
            return bot_reply
        else:
            return f"⚠️ Backend Error: {response.status_code}. Did you hit an API limit?"
            
    except requests.exceptions.ConnectionError:
        return "🚨 **Connection Error:** Could not reach the backend. Is your FastAPI server running on port 8000 with 'uvicorn main:app --reload'?"

# ============================================
# GRADIO UI SETUP
# ============================================

# We use a built-in modern theme for a premium look
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

# In Gradio 6.0, the ChatInterface is much simpler. 
# We just pass the function, text, and examples. It handles the buttons automatically!
demo = gr.ChatInterface(
    fn=chat_with_backend,
    title="Gemini AI Agent 🤖",
    description="Powered by FastAPI and Gradio. Ask me about the date, weather, or live news!",
    textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7),
    examples=[
        "What is today's date?",
        "What is the weather like in Tokyo?",
        "Give me the latest news about Artificial Intelligence."
    ]
)

# Launch the server!
if __name__ == "__main__":
    # In Gradio 6.0+, the theme parameter MUST be passed to launch(), not Blocks()
    demo.launch(share=False, theme=custom_theme)