from fastapi import FastAPI
from rag_system import process_rag_query, process_agent_query
from pydantic import BaseModel

app = FastAPI()

# Define the query model
class QueryModel(BaseModel):
    query: str

# RAG Endpoint
@app.post("/rag/")
async def rag_endpoint(query: QueryModel):
    response = process_rag_query(query.query)
    return {"response": response}

# Agent Endpoint with voice response
@app.post("/agent_with_voice/")
async def agent_endpoint(query: QueryModel):
    response = process_agent_query(query.query)
    return {"text_response": response["text_response"], "audio_url": response["audio_url"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
