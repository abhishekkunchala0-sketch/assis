"""Embedding and retrieval components for the agent."""
import time
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


class EmbeddingModel:
    """Wrapper for sentence transformer embeddings."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformer model
            device: Device to use (cpu, cuda)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        start_time = time.time()
        self._model = SentenceTransformer(self.model_name, device=self.device)
        elapsed = time.time() - start_time
        logger.info(f"Embedding model loaded in {elapsed:.2f}s")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self._model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        return self._model.encode(texts, convert_to_numpy=True, batch_size=32)
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._model.get_sentence_embedding_dimension()


class TableRetriever:
    """Retriever for database tables using semantic search."""
    
    def __init__(self, embedding_model: EmbeddingModel, schema_loader):
        """Initialize table retriever.
        
        Args:
            embedding_model: Embedding model instance
            schema_loader: Schema loader instance
        """
        self.embedding_model = embedding_model
        self.schema_loader = schema_loader
        self._index = None
        self._table_names: List[str] = []
        self._table_texts: List[str] = []
        self._build_index()
    
    def _build_index(self) -> None:
        """Build FAISS index from table metadata."""
        from faiss import IndexFlatL2, IndexFlatIP
        
        # Get all tables and their text representations
        tables = self.schema_loader.get_all_tables()
        self._table_names = [t.name for t in tables]
        self._table_texts = [t.to_text() for t in tables]
        
        # Create embeddings
        logger.info(f"Building index for {len(tables)} tables")
        start_time = time.time()
        embeddings = self.embedding_model.embed_texts(self._table_texts)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # Use inner product for cosine similarity
        dimension = embeddings.shape[1]
        self._index = IndexFlatIP(dimension)
        self._index.add(embeddings.astype('float32'))
        
        elapsed = time.time() - start_time
        logger.info(f"Index built in {elapsed:.2f}s")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Retrieve relevant tables for a query.
        
        Args:
            query: Natural language query
            top_k: Number of tables to retrieve
            threshold: Similarity threshold
            
        Returns:
            List of (table_name, similarity_score) tuples
        """
        # Embed query
        query_embedding = self.embedding_model.embed_text(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search index
        scores, indices = self._index.search(
            query_embedding.astype('float32').reshape(1, -1), 
            min(top_k, len(self._table_names))
        )
        
        # Filter by threshold and return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._table_names) and score >= threshold:
                results.append((self._table_names[idx], float(score)))
        
        # If no results above threshold, return top results anyway
        if not results:
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self._table_names):
                    results.append((self._table_names[idx], float(score)))
        
        return results[:top_k]
    
    def retrieve_with_context(
        self, 
        query: str, 
        top_k: int = 5, 
        threshold: float = 0.5
    ) -> Tuple[List[str], str]:
        """Retrieve tables and return context string.
        
        Args:
            query: Natural language query
            top_k: Number of tables to retrieve
            threshold: Similarity threshold
            
        Returns:
            Tuple of (table_names, context_string)
        """
        results = self.retrieve(query, top_k, threshold)
        table_names = [name for name, _ in results]
        context = self.schema_loader.get_table_context(table_names)
        
        return table_names, context
    
    def get_retrieval_quality_score(self, query: str, tables: List[str]) -> float:
        """Calculate retrieval quality score.
        
        Args:
            query: Original query
            tables: Retrieved table names
            
        Returns:
            Quality score (0-1)
        """
        if not tables:
            return 0.0
        
        # Check similarity scores
        results = self.retrieve(query, top_k=len(tables), threshold=0.0)
        result_dict = dict(results)
        
        # Average similarity of retrieved tables
        avg_score = np.mean([result_dict.get(t, 0) for t in tables])
        
        return min(avg_score, 1.0)
