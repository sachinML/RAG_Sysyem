# RAG_Sysyem

This project consists of three main parts:
1. **RAG System**: A system to retrieve relevant information from the NCERT PDF text using Vector Database and LLM.
2. **Smart Agent**: An agent that classifies user queries and decides when to use the RAG system or perform other tools like weather lookup or math evaluation.
3. **Voice Response**: Integration with Sarvam's Text-to-Speech (TTS) API to give voice to the agent's responses.

## Python Version
This project was developed and tested using **Python 3.10.0**. Ensure you are using this version or higher for compatibility.

## Setup

1. Clone the repository.
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Update the **API keys** for OpenAI and Sarvam in `rag.py`.
4. Run the FastAPI server:
    ```bash
    uvicorn server:app --reload
    ```
5. Run the Streamlit app for the RAG system:
    ```bash
    streamlit run main.py
    ```

## Endpoints

### 1. RAG Endpoint
- **URL**: `/rag/`
- **Method**: POST
- **Input**: A query related to NCERT content.
- **Output**: Relevant content from the NCERT PDF.

### 2. Agent with Voice Endpoint
- **URL**: `/agent_with_voice/`
- **Method**: POST
- **Input**: Any query.
- **Output**: The agent responds with both text and an audio file URL (using Sarvam’s API).

## Example Usage

### RAG Query:
```json
POST /rag/
{
  "query": "Explain the propagation of sound."
}
