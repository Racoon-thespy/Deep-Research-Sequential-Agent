Multi-Agent Market Research System

A Python-based multi-agent system for automated market research.
The system orchestrates multiple agents to collect, analyze, and summarize company and industry data using LLMs and external APIs.

Features

Multi-agent architecture:

Research Agent: Gathers data from Kaggle, HuggingFace, GitHub, and web sources.

Analysis Agent: Analyzes and structures the collected data.

Reporting Agent: Generates summaries and insights.

Orchestrator manages agent workflow and execution.

Streamlit interface for easy interaction.

Logging and monitoring of agent activities.


Installation

Clone the repo

git clone https://github.com/username/multi-agent-system.git
cd multi-agent-system


Create and activate a virtual environment

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Set up environment variables

Create a .env file in the root folder with your API keys:

OPENAI_API_KEY=your_openai_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
HUGGINGFACE_API_KEY=your_huggingface_key
GITHUB_TOKEN=your_github_token

Usage

Run the Streamlit app

streamlit run streamlit_app.py


Using the interface

Enter Company Name (e.g., Google)

Enter Industry (e.g., Technology)

Enter Additional Context / Description (optional)

Select Research Depth: quick, standard, deep

Click Start Research

Wait for the agents to finish and view results directly in the browser.
