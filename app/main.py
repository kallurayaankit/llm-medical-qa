from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import os

app = FastAPI()

# Load the vector store (created by prepare_rag.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Hugging Face Inference Client (free model)
# You can replace "mistralai/Mistral-7B-Instruct-v0.2" with any other small model
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2")

class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 200

@app.post("/ask")
def ask_medical_question(req: QueryRequest):
    # Retrieve relevant documents from Chroma
    docs = db.similarity_search(req.question, k=3)
    context = "\n".join([d.page_content for d in docs])

    # Build prompt
    prompt = f"<s>[INST] You are a helpful medical assistant. Use the context below to answer the question.\nContext: {context}\nQuestion: {req.question} [/INST]"

    # Call Hugging Face Inference API (free)
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=req.max_tokens,
            stop=["</s>"]
        )
        return {"answer": response, "sources": [d.metadata.get("source", "unknown") for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))