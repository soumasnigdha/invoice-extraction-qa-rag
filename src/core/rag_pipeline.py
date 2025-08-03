from langchain_community.llms import Ollama
from langchain.schema import Document
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import pandas as pd
import logging
import uuid

class RAGPipeline:
    def __init__(self):
        from ..config import Config
        from .vector_store import VectorStoreManager
        from ..prompts.rag_prompts import RAGPrompts
        
        self.config = Config()
        self.vector_store_manager = VectorStoreManager()
        self.llm = Ollama(
            model=self.config.OLLAMA_MODEL,
            base_url=self.config.OLLAMA_BASE_URL,
            temperature=0.1
        )
        self.prompts = RAGPrompts()
        self.logger = logging.getLogger(__name__)
    
    def index_document(self, 
                      file_path: Union[str, Path],
                      document_type: str = "invoice",
                      additional_metadata: Optional[Dict] = None) -> bool:
        """Index a document file in the vector store"""
        try:
            file_path = Path(file_path)
            
            # Generate document ID
            document_id = f"{file_path.stem}_{int(pd.Timestamp.now().timestamp())}"
            
            # Prepare metadata
            base_metadata = {
                "document_type": document_type,
                "indexed_at": str(pd.Timestamp.now()),
                "source_file": file_path.name
            }
            
            if additional_metadata:
                base_metadata.update(additional_metadata)
            
            # Process and add document to vector store
            chunk_ids = self.vector_store_manager.process_and_add_documents(file_path)
            
            if chunk_ids:
                self.logger.info(f"Successfully indexed document {document_id} with {len(chunk_ids)} chunks")
                return True
            else:
                self.logger.warning(f"No chunks created for document {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error indexing document: {e}")
            return False
    
    def extract_invoice_data_with_rag(self, 
                                     file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract invoice data using RAG approach"""
        try:
            from .document_processor import DocumentProcessor
            
            file_path = Path(file_path)
            document_id = f"{file_path.stem}_{int(pd.Timestamp.now().timestamp())}"
            
            # Load and process the current document
            processor = DocumentProcessor()
            documents = processor.load_document(file_path)
            
            if not documents:
                self.logger.error(f"Could not load document: {file_path}")
                return self._get_empty_invoice_structure()
            
            # Get the document text
            document_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Index the current document first
            self.index_document(file_path, document_type="invoice")
            
            # Get relevant context from similar invoices
            context = self.vector_store_manager.get_relevant_context(
                query=self._extract_key_features_for_search(document_text),
                document_type="invoice"
            )
            
            # Build context string
            context_str = self._build_context_string(context)
            
            # Create extraction prompt with context
            prompt = self.prompts.get_invoice_extraction_prompt()
            
            # Format prompt with context and current document
            formatted_prompt = prompt.format(
                context=context_str,
                current_document=document_text
            )
            
            # Get LLM response
            response = self.llm(formatted_prompt)
            
            # Parse JSON response
            extracted_data = self._parse_json_response(response)
            
            # Add metadata to extracted data
            extracted_data['_metadata'] = {
                'document_id': document_id,
                'source_file': file_path.name,
                'extraction_timestamp': str(pd.Timestamp.now()),
                'context_chunks_used': len(context)
            }
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error in RAG extraction: {e}")
            return self._get_empty_invoice_structure()
    
    def query_document_corpus(self, 
                            query: str,
                            document_type: Optional[str] = None,
                            top_k: int = 5) -> Dict[str, Any]:
        """Query the document corpus for specific information"""
        try:
            # Get relevant context
            context = self.vector_store_manager.get_relevant_context(
                query=query,
                document_type=document_type
            )
            
            if not context:
                return {
                    "query": query,
                    "response": "No relevant documents found in the corpus.",
                    "sources": [],
                    "total_relevant_chunks": 0
                }
            
            # Create query prompt
            prompt = self.prompts.get_query_prompt()
            context_str = self._build_context_string(context[:top_k])
            
            formatted_prompt = prompt.format(
                query=query,
                context=context_str
            )
            
            # Get LLM response
            response = self.llm(formatted_prompt)
            
            return {
                "query": query,
                "response": response,
                "sources": [ctx["metadata"] for ctx in context[:top_k]],
                "total_relevant_chunks": len(context)
            }
            
        except Exception as e:
            self.logger.error(f"Error in document corpus query: {e}")
            return {"error": str(e)}
    
    def find_similar_documents(self, 
                              reference_file: Union[str, Path],
                              document_type: Optional[str] = None,
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to the reference document"""
        try:
            from .document_processor import DocumentProcessor
            
            # Load reference document
            processor = DocumentProcessor()
            documents = processor.load_document(reference_file)
            
            if not documents:
                return []
            
            # Extract key features for similarity search
            document_text = "\n\n".join([doc.page_content for doc in documents])
            key_features = self._extract_key_features_for_search(document_text)
            
            # Search for similar documents
            similar_docs = self.vector_store_manager.similarity_search_with_scores(
                query=key_features,
                k=top_k * 2,  # Get more for filtering
                filter_metadata={"document_type": document_type} if document_type else None
            )
            
            # Process and rank results
            results = []
            seen_documents = set()
            
            for doc, score in similar_docs:
                # Avoid duplicates from the same document
                doc_id = doc.metadata.get('document_id', 'unknown')
                if doc_id in seen_documents:
                    continue
                
                seen_documents.add(doc_id)
                
                results.append({
                    "document_id": doc_id,
                    "source_file": doc.metadata.get('source_file', 'unknown'),
                    "content_preview": doc.page_content[:300] + "...",
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar documents: {e}")
            return []
    
    def analyze_document_patterns(self, document_type: str = "invoice") -> Dict[str, Any]:
        """Analyze patterns across documents of a specific type"""
        try:
            # Get collection statistics
            stats = self.vector_store_manager.get_collection_stats()
            
            # Query for common patterns
            pattern_queries = [
                "vendor name supplier company",
                "total amount invoice value",
                "tax GST CGST SGST",
                "payment terms due date",
                "invoice number format"
            ]
            
            patterns = {}
            for query in pattern_queries:
                results = self.query_document_corpus(
                    query=query,
                    document_type=document_type,
                    top_k=3
                )
                patterns[query] = results
            
            return {
                "document_type": document_type,
                "total_documents": stats.get('unique_documents', 0),
                "total_chunks": stats.get('total_chunks', 0),
                "patterns_found": patterns,
                "analysis_timestamp": str(pd.Timestamp.now())
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing document patterns: {e}")
            return {"error": str(e)}
    
    def _extract_key_features_for_search(self, document_text: str) -> str:
        """Extract key features from document for similarity search"""
        lines = document_text.split('\n')
        key_lines = []
        
        # Keywords to look for
        keywords = [
            'invoice', 'bill', 'amount', 'total', 'vendor', 'supplier',
            'customer', 'tax', 'gst', 'payment', 'date', 'number'
        ]
        
        for line in lines:
            line_lower = line.strip().lower()
            if any(keyword in line_lower for keyword in keywords):
                key_lines.append(line.strip())
        
        # Limit to prevent huge queries
        return ' '.join(key_lines[:15])
    
    def _build_context_string(self, context: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        if not context:
            return "No relevant context found."
        
        context_parts = []
        for i, ctx in enumerate(context):
            source_info = ctx['metadata'].get('source_file', 'Unknown')
            relevance = ctx.get('relevance_score', 0)
            
            context_parts.append(
                f"Context {i+1} (Source: {source_info}, Relevance: {relevance:.3f}):\n"
                f"{ctx['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        import re
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                self.logger.warning("No valid JSON found in LLM response")
                return self._get_empty_invoice_structure()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {e}")
            return self._get_empty_invoice_structure()
    
    def _get_empty_invoice_structure(self) -> Dict[str, Any]:
        """Return empty invoice structure for error cases"""
        return {
            "invoice_info": {
                "invoice_number": "",
                "invoice_date": "",
                "due_date": "",
                "po_number": "",
                "currency": ""
            },
            "vendor": {
                "name": "",
                "address": "",
                "phone": "",
                "email": "",
                "gst_number": "",
                "pan": ""
            },
            "customer": {
                "name": "",
                "address": "",
                "phone": "",
                "email": "",
                "gst_number": "",
                "pan": ""
            },
            "line_items": [],
            "totals": {
                "tax_total": 0,
                "grand_total": 0
            },
            "payment_info": {
                "payment_terms": "",
                "payment_method": "",
                "bank_account": "",
                "ifsc_code": ""
            }
        }
    
    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document corpus"""
        try:
            stats = self.vector_store_manager.get_collection_stats()
            
            # Add RAG-specific statistics
            stats['rag_pipeline_info'] = {
                'model': self.config.OLLAMA_MODEL,
                'chunk_size': self.config.CHUNK_SIZE,
                'similarity_threshold': self.config.SIMILARITY_THRESHOLD,
                'max_relevant_chunks': self.config.MAX_RELEVANT_CHUNKS
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting corpus statistics: {e}")
            return {"error": str(e)}
