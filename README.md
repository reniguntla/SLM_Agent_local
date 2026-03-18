# Local SLM Chat Application

A Streamlit chat interface for interacting with Small Language Models (SLMs) running locally in Ollama.

## Features

- Select a supported SLM from the sidebar.
- Phi-3 is the default model.
- Conversation memory keeps the last 10 question-answer interactions.
- Streaming responses from Ollama for a more interactive chat experience.
- Clear conversation button to reset chat history.
- Basic context trimming to keep requests within a reasonable token budget.
- Status messaging to show which model is currently in use and whether Phi-3 is already running.

## Supported models

The application is structured to support the following Ollama models:

- `phi3` (default)
- `gemma`
- `mistral`

The sidebar automatically shows supported models that are already installed in the local Ollama instance.

## Run locally

1. Ensure Ollama is installed and running locally.
2. Pull Phi-3 if needed:

   ```bash
   ollama pull phi3
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the app:

   ```bash
   streamlit run app.py
   ```

The app expects Ollama at `http://localhost:11434`.
