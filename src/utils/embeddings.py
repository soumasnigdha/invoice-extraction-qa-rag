from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
import os
import logging

class EmbeddingManager:
    def __init__(self, model_name: Optional[str] = None):
        from ..config import Config
        self.config = Config()
        self.model_name = model_name or self.config.EMBEDDING_MODEL
        self.embeddings = self._initialize_embeddings()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            if "openai" in self.model_name.lower():
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self.logger.warning("OpenAI API key not found, falling back to HuggingFace")
                    return self._get_huggingface_embeddings()
                
                return OpenAIEmbeddings(
                    model=self.model_name,
                    api_key=api_key
                )
            else:
                return self._get_huggingface_embeddings()
                
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            return self._get_fallback_embeddings()
    
    def _get_huggingface_embeddings(self):
        """Get HuggingFace embeddings with updated import"""
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _get_fallback_embeddings(self):
        """Get fallback embeddings"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def get_embeddings(self):
        """Get the embedding model instance"""
        return self.embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string"""
        try:
            if not text or not text.strip():
                self.logger.warning("Empty text provided for embedding")
                return []
            return self.embeddings.embed_query(text)
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            return []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            if not texts:
                return []
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                self.logger.warning("No valid texts provided for embedding")
                return []
                
            return self.embeddings.embed_documents(valid_texts)
        except Exception as e:
            self.logger.error(f"Error embedding documents: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        try:
            test_embedding = self.embed_text("test")
            return len(test_embedding) if test_embedding else self.config.EMBEDDING_DIMENSION
        except Exception:
            return self.config.EMBEDDING_DIMENSION
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            emb1 = self.embed_text(text1)
            emb2 = self.embed_text(text2)
            
            if not emb1 or not emb2:
                return 0.0
            
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
