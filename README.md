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

## ⚙️ Installation & Setup

Before running this application, you must configure your Python environment and API keys.

### 1. Prerequisites

* **Python:** Version 3.8 or higher installed on your machine.
* **APIs Used (Keys Required):** You will need to sign up for free API keys from the following services:
  * [Google AI Studio](https://aistudio.google.com/) (for the Gemini LLM)
  * [OpenWeatherMap](https://openweathermap.org/api) (for live weather data)
  * [NewsData.io](https://newsdata.io/) (for live news headlines)


