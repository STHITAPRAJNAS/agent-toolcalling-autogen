from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from .config import MemoryConfig
from loguru import logger
import json

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'))
    role = Column(String)  # user, assistant, system, function
    content = Column(String)
    function_call = Column(JSON, nullable=True)
    function_response = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

class ConversationMemory:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.engine = self._create_engine()
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _create_engine(self):
        """Create database engine based on configuration."""
        if self.config.type == "sqlite":
            return create_engine(f"sqlite:///{self.config.path}")
        else:  # postgresql
            return create_engine(
                f"postgresql://{self.config.connection['user']}:{self.config.connection['password']}@"
                f"{self.config.connection['host']}:{self.config.connection['port']}/"
                f"{self.config.connection['database']}"
            )

    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a given conversation ID."""
        try:
            with self.Session() as session:
                conversation = session.query(Conversation).filter_by(id=conversation_id).first()
                if not conversation:
                    return []
                
                messages = session.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
                return [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "function_call": msg.function_call,
                        "function_response": msg.function_response,
                        "timestamp": msg.timestamp.isoformat()
                    }
                    for msg in messages
                ]
                
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []

    async def add_to_history(
        self,
        conversation_id: str,
        user_id: str,
        message: Dict[str, Any]
    ) -> None:
        """Add a message to conversation history."""
        try:
            with self.Session() as session:
                # Get or create conversation
                conversation = session.query(Conversation).filter_by(id=conversation_id).first()
                if not conversation:
                    conversation = Conversation(id=conversation_id, user_id=user_id)
                    session.add(conversation)
                
                # Create message
                msg = Message(
                    conversation_id=conversation_id,
                    role=message["role"],
                    content=message.get("content"),
                    function_call=message.get("function_call"),
                    function_response=message.get("function_response")
                )
                session.add(msg)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error adding to conversation history: {str(e)}")
            raise

    async def clear_history(self, conversation_id: str) -> None:
        """Clear conversation history for a given conversation ID."""
        try:
            with self.Session() as session:
                session.query(Message).filter_by(conversation_id=conversation_id).delete()
                session.query(Conversation).filter_by(id=conversation_id).delete()
                session.commit()
                
        except Exception as e:
            logger.error(f"Error clearing conversation history: {str(e)}")
            raise

    async def get_recent_conversations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations for a user."""
        try:
            with self.Session() as session:
                conversations = (
                    session.query(Conversation)
                    .filter_by(user_id=user_id)
                    .order_by(Conversation.updated_at.desc())
                    .limit(limit)
                    .all()
                )
                
                result = []
                for conv in conversations:
                    last_message = (
                        session.query(Message)
                        .filter_by(conversation_id=conv.id)
                        .order_by(Message.timestamp.desc())
                        .first()
                    )
                    
                    if last_message:
                        result.append({
                            "conversation_id": conv.id,
                            "last_message": {
                                "role": last_message.role,
                                "content": last_message.content,
                                "timestamp": last_message.timestamp.isoformat()
                            },
                            "created_at": conv.created_at.isoformat(),
                            "updated_at": conv.updated_at.isoformat()
                        })
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting recent conversations: {str(e)}")
            return []

    async def search_history(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search through conversation history."""
        try:
            with self.Session() as session:
                # Build query
                search_query = session.query(Message).join(Conversation)
                if user_id:
                    search_query = search_query.filter(Conversation.user_id == user_id)
                
                # Search in content
                search_query = search_query.filter(
                    Message.content.ilike(f"%{query}%")
                )
                
                # Get results
                messages = search_query.order_by(Message.timestamp.desc()).limit(limit).all()
                
                return [
                    {
                        "conversation_id": msg.conversation_id,
                        "user_id": msg.conversation.user_id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat()
                    }
                    for msg in messages
                ]
                
        except Exception as e:
            logger.error(f"Error searching conversation history: {str(e)}")
            return [] 