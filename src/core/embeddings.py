from typing import Dict, Any, List, Optional
import numpy as np
from abc import ABC, abstractmethod
from .config import EmbeddingConfig
from loguru import logger
import psycopg2
from psycopg2.extras import Json
import json

class EmbeddingStore(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        pass

    @abstractmethod
    async def store_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Store embeddings with metadata."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        pass

class MemoryEmbeddingStore(EmbeddingStore):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embeddings = []
        self.metadata = []

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using the configured LLM."""
        # In a real implementation, this would call the LLM's embedding API
        # For now, return a random embedding
        return np.random.randn(self.config.dimension)

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        return [await self.get_embedding(text) for text in texts]

    async def store_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Store embeddings with metadata in memory."""
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in memory."""
        if not self.embeddings:
            return []

        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Apply filter if provided
        if filter:
            mask = np.ones(len(self.metadata), dtype=bool)
            for key, value in filter.items():
                mask &= np.array([m.get(key) == value for m in self.metadata])
            similarities[~mask] = -1

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                "content": self.metadata[i].get("content", ""),
                "metadata": self.metadata[i],
                "score": float(similarities[i])
            }
            for i in top_indices
            if similarities[i] > 0
        ]

class PGVectorEmbeddingStore(EmbeddingStore):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.conn = self._create_connection()

    def _create_connection(self) -> psycopg2.extensions.connection:
        """Create PostgreSQL connection."""
        return psycopg2.connect(
            host=self.config.connection["host"],
            port=self.config.connection["port"],
            database=self.config.connection["database"],
            user=self.config.connection["user"],
            password=self.config.connection["password"]
        )

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using the configured LLM."""
        # In a real implementation, this would call the LLM's embedding API
        # For now, return a random embedding
        return np.random.randn(self.config.dimension)

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        return [await self.get_embedding(text) for text in texts]

    async def store_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Store embeddings with metadata in PostgreSQL."""
        try:
            with self.conn.cursor() as cur:
                for embedding, meta in zip(embeddings, metadata):
                    cur.execute(
                        """
                        INSERT INTO embeddings (embedding, metadata)
                        VALUES (%s, %s)
                        """,
                        (embedding.tolist(), Json(meta))
                    )
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing embeddings: {str(e)}")
            raise

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in PostgreSQL."""
        try:
            with self.conn.cursor() as cur:
                # Build filter condition
                filter_condition = ""
                if filter:
                    conditions = []
                    for key, value in filter.items():
                        conditions.append(f"metadata->>'{key}' = %s")
                    filter_condition = "WHERE " + " AND ".join(conditions)

                # Execute search query
                cur.execute(
                    f"""
                    SELECT metadata, 1 - (embedding <=> %s) as similarity
                    FROM embeddings
                    {filter_condition}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), top_k)
                )
                
                results = cur.fetchall()
                
                return [
                    {
                        "content": result[0].get("content", ""),
                        "metadata": result[0],
                        "score": float(result[1])
                    }
                    for result in results
                ]
                
        except Exception as e:
            logger.error(f"Error searching embeddings: {str(e)}")
            raise

def get_embedding_store(config: EmbeddingConfig) -> EmbeddingStore:
    """Get the appropriate embedding store based on configuration."""
    if config.store == "memory":
        return MemoryEmbeddingStore(config)
    elif config.store == "pgvector":
        return PGVectorEmbeddingStore(config)
    else:
        raise ValueError(f"Unsupported embedding store: {config.store}") 