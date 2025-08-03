# src/core/vector_store.py
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import logging  # ← Make sure this import is present
import uuid
from pathlib import Path

class VectorStoreManager:
    def __init__(self):
        from ..config import Config
        from ..utils.embeddings import EmbeddingManager
        
        self.config = Config()
        self.embedding_manager = EmbeddingManager()
        self.embeddings = self.embedding_manager.get_embeddings()
        
        # Initialize logger BEFORE calling any methods that use it
        self.logger = logging.getLogger(__name__)
        
        # Now initialize vector store (which may use self.logger)
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize the vector database"""
        try:
            # Rest of your method...
            persist_dir = Path(self.config.CHROMA_PERSIST_DIR)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            vector_store = Chroma(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir)
            )
            
            self.logger.info(f"Vector store initialized: {self.config.COLLECTION_NAME}")
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize the vector database"""
        try:
            # Ensure the persist directory exists
            persist_dir = Path(self.config.CHROMA_PERSIST_DIR)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB with persistence
            vector_store = Chroma(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir)
            )
            
            self.logger.info(f"Vector store initialized: {self.config.COLLECTION_NAME}")
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add multiple Document objects to the vector store"""
        try:
            if not documents:
                self.logger.warning("No documents provided to add")
                return []
            
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_documents:
                self.logger.warning("No valid documents to add")
                return []
            
            # Generate unique IDs for documents if not present
            document_ids = []
            for i, doc in enumerate(valid_documents):
                if 'chunk_id' not in doc.metadata:
                    doc_id = doc.metadata.get('document_id', f"doc_{uuid.uuid4().hex[:8]}")
                    chunk_id = f"{doc_id}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                    doc.metadata['chunk_id'] = chunk_id
                
                document_ids.append(doc.metadata['chunk_id'])
                
                # Add timestamp if not present
                if 'timestamp' not in doc.metadata:
                    doc.metadata['timestamp'] = str(pd.Timestamp.now())
            
            # Add to vector store
            added_ids = self.vector_store.add_documents(valid_documents, ids=document_ids)
            
            # Persist the changes
            self.vector_store.persist()
            
            self.logger.info(f"Added {len(valid_documents)} documents to vector store")
            return added_ids
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def add_document(self, 
                    document_text: str, 
                    metadata: Dict[str, Any],
                    document_id: str) -> List[str]:
        """Add a single document (backward compatibility)"""
        try:
            if not document_text or not document_text.strip():
                self.logger.warning(f"Empty document text for document_id: {document_id}")
                return []
            
            # Create Document object
            doc = Document(
                page_content=document_text,
                metadata={
                    **metadata,
                    "document_id": document_id,
                    "timestamp": str(pd.Timestamp.now()),
                    "text_length": len(document_text)
                }
            )
            
            return self.add_documents([doc])
            
        except Exception as e:
            self.logger.error(f"Error adding document to vector store: {e}")
            raise
    
    def process_and_add_documents(self, file_path: Union[str, Path]) -> List[str]:
        """Process a document file and add it to vector store"""
        try:
            from .document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            
            # Load and split documents
            documents = processor.load_document(file_path)
            if not documents:
                self.logger.warning(f"No documents loaded from {file_path}")
                return []
            
            # Split into chunks
            chunks = processor.split_documents(documents)
            
            # Add to vector store
            return self.add_documents(chunks)
            
        except Exception as e:
            self.logger.error(f"Error processing and adding document {file_path}: {e}")
            return []
    
    def similarity_search(self, 
                         query: str, 
                         k: int = None,
                         filter_metadata: Optional[Dict] = None,
                         score_threshold: float = None) -> List[Document]:
        """Perform similarity search"""
        if k is None:
            k = self.config.MAX_RELEVANT_CHUNKS
        
        try:
            if not query or not query.strip():
                self.logger.warning("Empty query provided for similarity search")
                return []
            
            # Use similarity search with score if threshold provided
            if score_threshold is not None:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
                # Filter by score threshold
                filtered_results = [
                    doc for doc, score in results 
                    if score >= score_threshold
                ]
                return filtered_results
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
                return results
                
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_scores(self,
                                    query: str,
                                    k: int = None,
                                    filter_metadata: Optional[Dict] = None) -> List[tuple]:
        """Perform similarity search with scores"""
        if k is None:
            k = self.config.MAX_RELEVANT_CHUNKS
        
        try:
            if not query or not query.strip():
                return []
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search with scores: {e}")
            return []
    
    def get_relevant_context(self, 
                           query: str,
                           document_type: Optional[str] = None,
                           min_score: float = None) -> List[Dict[str, Any]]:
        """Get relevant context for a query"""
        # Build filter if document type specified
        filter_metadata = None
        if document_type:
            filter_metadata = {"document_type": document_type}
        
        # Perform similarity search with scores
        results = self.similarity_search_with_scores(
            query=query,
            filter_metadata=filter_metadata
        )
        
        # Format results
        context = []
        for doc, score in results:
            # Skip if below minimum score threshold
            if min_score is not None and score < min_score:
                continue
                
            context.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            })
        
        return context
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks from vector store"""
        try:
            # Get the collection directly for more control
            collection = self.vector_store._collection
            
            # Query for chunks belonging to this document
            results = collection.get(
                where={"document_id": document_id}
            )
            
            if results and results['ids']:
                # Delete the chunks
                collection.delete(ids=results['ids'])
                self.vector_store.persist()
                
                self.logger.info(f"Deleted document {document_id} and {len(results['ids'])} chunks")
                return True
            else:
                self.logger.warning(f"No chunks found for document_id: {document_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get unique document IDs and file types
            all_metadata = collection.get()
            unique_docs = set()
            document_types = {}
            file_types = {}
            
            if all_metadata and all_metadata['metadatas']:
                for metadata in all_metadata['metadatas']:
                    if 'document_id' in metadata:
                        unique_docs.add(metadata['document_id'])
                    
                    if 'document_type' in metadata:
                        doc_type = metadata['document_type']
                        document_types[doc_type] = document_types.get(doc_type, 0) + 1
                    
                    if 'file_type' in metadata:
                        file_type = metadata['file_type']
                        file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                "total_chunks": count,
                "unique_documents": len(unique_docs),
                "document_types": document_types,
                "file_types": file_types,
                "collection_name": self.config.COLLECTION_NAME,
                "embedding_dimension": self.embedding_manager.get_embedding_dimension()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                "total_chunks": 0,
                "unique_documents": 0,
                "document_types": {},
                "file_types": {},
                "collection_name": self.config.COLLECTION_NAME,
                "error": str(e)
            }
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            collection = self.vector_store._collection
            all_data = collection.get()
            
            if all_data and all_data['ids']:
                collection.delete(ids=all_data['ids'])
                self.vector_store.persist()
                self.logger.info(f"Cleared {len(all_data['ids'])} chunks from collection")
                return True
            else:
                self.logger.info("Collection was already empty")
                return True
                
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False
