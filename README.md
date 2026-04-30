# 🤖 Gemini AI Chat: Backend & Python Frontends

Welcome to the **Gemini AI Chat Backend** repository! 

This repository serves as the core intelligence hub for a modular AI ecosystem powered by Google's **Gemini AI**. It features a robust **FastAPI** backend that acts as the "brain," enhanced with **Function Calling (Tooling)** to interact with real-world data APIs. 

This repo also includes two distinct Python-based web interfaces (**Streamlit** and **Gradio**) to demonstrate how multiple frontend "faces" can connect to a single unified backend architecture.

*(Note: The React frontend for this project is hosted in a separate repository).*

---

## ✨ Key Features

### 🧠 The Backend (FastAPI)
* **RESTful Architecture:** Built with FastAPI for high performance, data validation (Pydantic), and automatic documentation generation (Swagger/OpenAPI).
* **CORS Enabled:** Fully configured to accept Cross-Origin requests, making it compatible with modern JavaScript external frontends (like React, Vue, or Angular).
* **Gemini Tool Calling:** The AI model is equipped with custom Python functions to fetch live data dynamically based on the user's prompt:
  * 📅 **Live Date & Time**
  * 🌤️ **Live Weather** (via OpenWeatherMap API)
  * 📰 **Live News** (via NewsData.io API)

### 🎨 The Frontends
This repository contains two integrated Python UI implementations that connect seamlessly to the FastAPI backend:
1. **Streamlit UI (`frontend.py`):** A sophisticated interface demonstrating session state management, chat memory, KPIs (`st.metric`), and complex layouts.
2. **Gradio UI (`gradio_ui.py`):** A clean, modern chat interface showcasing how rapidly machine learning models can be deployed using the industry-standard Gradio framework.

---

## 📂 Project Structure

```text
gemini-cli-chat/
│
├── main.py              # The FastAPI backend server & AI agent logic
├── frontend.py          # The Streamlit web interface
├── gradio_ui.py         # The Gradio web interface
├── requirements.txt     # Python dependencies
└── .env                 # API Keys (Not tracked in git)


⚙️ Installation & Setup
Before running this application, you must configure your Python environment and API keys.

1. Prerequisites
Python: Version 3.8 or higher installed on your machine.

API Keys: You will need to sign up for free keys from:

Google AI Studio (for Gemini)

OpenWeatherMap (for Weather)

NewsData.io (for News)

2. Clone the Repository
Bash
git clone [https://github.com/Drutika06/Gemini-ChatBackend.git](https://github.com/Drutika06/Gemini-ChatBackend.git)
cd Gemini-ChatBackend/gemini-cli-chat
3. Setup Virtual Environment & Install Dependencies
It is highly recommended to use a virtual environment to avoid dependency conflicts.

Bash
# Create the environment
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate

# Activate the environment (Mac/Linux)
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt
4. Configure Environment Variables
Create a file named exactly .env in the root of the gemini-cli-chat folder and paste your keys:

Code snippet
GEMINI_API_KEY="your_actual_gemini_api_key_here"
OPENWEATHER_API_KEY="your_actual_openweathermap_key_here"
NEWS_API_KEY="your_actual_newsdata_key_here"
💻 How to Run the Application
This project uses a decoupled architecture. You must run the backend server and your chosen frontend simultaneously in two separate terminal windows.

Step 1: Start the Backend (Terminal 1)
This command boots up the "brain" and tools on port 8000.

Bash
uvicorn main:app --reload
Verify Online: http://127.0.0.1:8000/

Interactive Docs: http://127.0.0.1:8000/docs

Step 2: Start a Frontend (Terminal 2)
Open a brand new terminal, ensure your virtual environment is active, and choose which "face" to run:

Option A: Run Streamlit

Bash
streamlit run frontend.py
(Automatically opens at http://localhost:8501)

Option B: Run Gradio

Bash
python gradio_ui.py
(Provides a local link, usually http://127.0.0.1:7860)

🛠️ How It Works: Architecture Overview
This application demonstrates a clean Client-Server communication model.

The Data Flow
User Input: A user types a prompt into the frontend (Streamlit, Gradio, or React).

API Request: The frontend packages the prompt into a JSON object {"message": "..."} and sends an HTTP POST request to http://127.0.0.1:8000/chat.

Backend Processing:

FastAPI receives the message and passes it to the Gemini session (chat_session.send_message).

Gemini analyzes the prompt and determines if it needs one of the custom Python tools (e.g., if the user asks for weather).

If a tool is needed, Gemini calls that Python function in main.py automatically.

The function fetches live data (e.g., from OpenWeatherMap) and returns it to Gemini.

Gemini uses the raw data to formulate a natural language response.

API Response: FastAPI receives the final response text from Gemini and sends it back to the frontend in a structured JSON format {"reply": "..."}.

Display: The frontend UI extracts the reply and updates the screen for the user.


The FastAPI backend includes CORSMiddleware configured to allow_origins=["*"]. This is what allows your React application (running on a different port) or even a mobile app to talk to this server without being blocked by the browser's security policies.
