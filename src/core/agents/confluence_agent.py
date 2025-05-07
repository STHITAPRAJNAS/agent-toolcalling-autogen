from typing import Dict, Any, List, Optional
import numpy as np
from .base_agent import BaseAgent
from ..embeddings import get_embedding_store
from ..config import Config, AgentConfig

class ConfluenceAgent(BaseAgent):
    def __init__(self, config: Config, agent_config: AgentConfig):
        super().__init__(config, agent_config)
        self.embedding_store = get_embedding_store(config.embedding)
        self.batch_size = agent_config.embedding_batch_size or 100

    def _get_system_message(self) -> str:
        return f"""You are {self.agent_config.name}, {self.agent_config.description}.
        You are an expert at retrieving and synthesizing information from Confluence pages.
        You have access to a vector database containing Confluence page embeddings.
        When answering questions:
        1. First, search for relevant information in the vector database
        2. Use the retrieved context to provide accurate and comprehensive answers
        3. If you're unsure about something, acknowledge the uncertainty
        4. Always cite your sources when possible
        5. Keep responses concise but informative"""

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Get relevant documents from the embedding store
            relevant_docs = await self._get_relevant_documents(message)
            
            # Prepare context with retrieved documents
            enhanced_context = {
                "retrieved_documents": relevant_docs,
                **(context or {})
            }

            # Process the message with enhanced context
            return await super().process_message(message, enhanced_context)

        except Exception as e:
            return await self.handle_error(e)

    async def _get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the embedding store."""
        # Get query embedding
        query_embedding = await self.embedding_store.get_embedding(query)
        
        # Search for similar documents
        results = await self.embedding_store.search(
            query_embedding,
            top_k=top_k,
            filter={"source": "confluence"}
        )
        
        return [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": doc["score"]
            }
            for doc in results
        ]

    async def update_embeddings(self, documents: List[Dict[str, Any]]):
        """Update the embedding store with new documents."""
        try:
            # Process documents in batches
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                
                # Get embeddings for the batch
                embeddings = await self.embedding_store.get_embeddings(
                    [doc["content"] for doc in batch]
                )
                
                # Store embeddings with metadata
                await self.embedding_store.store_embeddings(
                    embeddings,
                    [doc["metadata"] for doc in batch]
                )
                
        except Exception as e:
            logger.error(f"Error updating embeddings: {str(e)}")
            raise 