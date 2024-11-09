# Content Engine

This project is a Streamlit-based application that creates a content engine using Retrieval Augmented Generation (RAG) techniques to provide answers from embedded data. This application supports interactive chat functionality to retrieve and respond to user queries based on document embeddings.

## Prerequisites

1. **Python Version**: Requires Python 3.9
2. **Ollama Installation**: This code uses the local LLM model from Ollama; install Ollama following instructions at [Ollama's official site](https://ollama.com). Ensure the `gemma2:2b` model is downloaded and configured.
3. **Dependencies**: Install required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

## Initial Setup

1. **Run Ollama**:
   - Start the Ollama service and ensure the `gemma2:2b` model is available. Run:
     ```bash
     ollama run gemma2:2b
     ```

2. **Vector Database Creation**:
   - Execute the `vector_database_creation` notebook to create the initial vector store. This generates the `vector_store` in the `Data` directory.

## Usage

1. **Start Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
   This will open a local web server where you can interact with the content engine.

2. **Using the Application**:
   - Enter questions in the chat interface. The application retrieves relevant information based on vectorized document embeddings and responds with generated answers.

## Code Walkthrough

- **Embedding Model**: Uses `sentence-transformers/all-mpnet-base-v2` for document embeddings.
- **Local LLM**: Uses the Ollama `gemma2:2b` model to process questions and generate answers.
- **Vector Store Loading**: The app checks for an existing vector store, and if absent, loads the pre-existing one in the `Data` folder. Ensure `vector_database_creation.py` has been run to generate the initial embeddings.
- **Retrieval Chain**: Utilizes FAISS to retrieve relevant content and provides responses based on pre-trained document embeddings.

