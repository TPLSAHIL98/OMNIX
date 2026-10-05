from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from omnix_engine import OMNIXEngine


app = FastAPI(
    title="OMNIX API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


engine = OMNIXEngine()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "name": "OMNIX",
        "version": "1.0.0",
        "status": "online",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "OMNIX 1.0",
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        return {
            "response": "Please enter a message."
        }

    response = engine.generate(
        message
    )

    return {
        "response": response
    }
