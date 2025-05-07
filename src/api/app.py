from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
from ..core.config import get_config
from ..core.agents.supervisor_agent import SupervisorAgent
from ..core.memory import ConversationMemory
from loguru import logger
import time

app = FastAPI(
    title="Agentic Knowledge System",
    description="A system of specialized agents for knowledge retrieval and query processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = get_config()

# Initialize agents and memory
supervisor = SupervisorAgent(config, config.agents["supervisor"])
memory = ConversationMemory(config.memory)

class MessageRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    conversation_id: str

class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Process a chat message and return a response."""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Process message with supervisor agent
        response = await supervisor.process_message(
            request.message,
            {
                "conversation_id": conversation_id,
                "user_id": request.user_id
            }
        )
        
        return MessageResponse(
            response=response,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{user_id}", response_model=List[ConversationHistory])
async def get_conversations(user_id: str, limit: int = 10):
    """Get recent conversations for a user."""
    try:
        conversations = await memory.get_recent_conversations(user_id, limit)
        return conversations
        
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/history", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """Get conversation history for a specific conversation."""
    try:
        history = await memory.get_conversation_history(conversation_id)
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=history
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history for a specific conversation."""
    try:
        await memory.clear_history(conversation_id)
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=List[Dict[str, Any]])
async def search_history(
    query: str,
    user_id: Optional[str] = None,
    limit: int = 10
):
    """Search through conversation history."""
    try:
        results = await memory.search_history(query, user_id, limit)
        return results
        
    except Exception as e:
        logger.error(f"Error searching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    } 