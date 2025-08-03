import sys
from pathlib import Path
import logging
import pandas as pd

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.document_processor import DocumentProcessor
from src.core.rag_pipeline import RAGPipeline
from src.core.excel_handler import ExcelHandler
from src.core.template_mapper import TemplateMapper
from src.config import Config

from typing import Union, Dict, Any, List
import json

class FinancialRAGApp:
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor()
        self.rag_pipeline = RAGPipeline()
        self.excel_handler = ExcelHandler()
        self.template_mapper = TemplateMapper()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Financial RAG Application initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.config.LOGS_DIR / "app.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single financial document using RAG approach"""
        try:
            file_path = Path(file_path)
            self.logger.info(f"Processing document with RAG: {file_path.name}")
            
            # Validate document
            if not self.document_processor.validate_document(file_path):
                raise ValueError(f"Unsupported document format: {file_path.suffix}")
            
            # Extract structured data using RAG
            self.logger.info("Extracting structured data with RAG...")
            invoice_data = self.rag_pipeline.extract_invoice_data_with_rag(file_path)
            
            # Get invoice number for naming
            invoice_number = invoice_data.get('invoice_info', {}).get('invoice_number', f"UNKNOWN_{int(pd.Timestamp.now().timestamp())}")
            
            # Save individual invoice Excel
            self.logger.info("Saving individual invoice Excel...")
            individual_excel_path = self.excel_handler.save_individual_invoice(
                invoice_data, invoice_number
            )
            
            # Map to master template
            self.logger.info("Mapping to master template...")
            mapped_data = self.template_mapper.map_to_template(invoice_data)
            
            # Validate mapped data
            validation_results = self.template_mapper.validate_mapped_data(mapped_data)
            
            # Update master Excel
            self.logger.info("Updating master Excel...")
            master_excel_path = self.excel_handler.update_master_excel(mapped_data)
            
            # Get mapping summary
            mapping_summary = self.template_mapper.get_mapping_summary(invoice_data, mapped_data)
            
            result = {
                'status': 'success',
                'file_path': str(file_path),
                'individual_excel': individual_excel_path,
                'master_excel': master_excel_path,
                'invoice_number': invoice_number,
                'extracted_data': invoice_data,
                'mapped_data': mapped_data,
                'validation_results': validation_results,
                'mapping_summary': mapping_summary,
                'processing_timestamp': str(pd.Timestamp.now())
            }
            
            self.logger.info(f"Successfully processed {file_path.name} with RAG")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'file_path': str(file_path),
                'processing_timestamp': str(pd.Timestamp.now())
            }
    
    def process_batch(self, directory_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """Process all supported documents in input directory"""
        if directory_path is None:
            directory_path = self.config.INVOICES_DIR
        
        directory_path = Path(directory_path)
        results = []
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        # Get all supported files
        supported_files = []
        for ext in self.document_processor.supported_formats:
            supported_files.extend(directory_path.glob(f"*{ext}"))
        
        self.logger.info(f"Found {len(supported_files)} supported files for batch processing")
        
        for file_path in supported_files:
            result = self.process_document(file_path)
            results.append(result)
            
            # Log progress
            processed = len(results)
            self.logger.info(f"Batch progress: {processed}/{len(supported_files)} files processed")
        
        return results
    
    def query_documents(self, query: str, document_type: str = None) -> Dict[str, Any]:
        """Query the indexed document corpus"""
        try:
            result = self.rag_pipeline.query_document_corpus(
                query=query,
                document_type=document_type
            )
            self.logger.info(f"Query executed: '{query}' - Found {result.get('total_relevant_chunks', 0)} relevant chunks")
            return result
        except Exception as e:
            self.logger.error(f"Error querying documents: {e}")
            return {"error": str(e)}
    
    def find_similar_invoices(self, reference_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """Find invoices similar to the reference file"""
        try:
            similar_docs = self.rag_pipeline.find_similar_documents(
                reference_file=reference_file,
                document_type="invoice"
            )
            self.logger.info(f"Found {len(similar_docs)} similar invoices to {Path(reference_file).name}")
            return similar_docs
        except Exception as e:
            self.logger.error(f"Error finding similar invoices: {e}")
            return []
    
    def analyze_document_patterns(self, document_type: str = "invoice") -> Dict[str, Any]:
        """Analyze patterns across documents"""
        try:
            patterns = self.rag_pipeline.analyze_document_patterns(document_type)
            self.logger.info(f"Pattern analysis completed for document type: {document_type}")
            return patterns
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
            return {"error": str(e)}
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the document corpus"""
        try:
            vector_stats = self.rag_pipeline.get_corpus_statistics()
            excel_stats = self.excel_handler.get_master_excel_stats()
            
            return {
                "vector_database": vector_stats,
                "master_excel": excel_stats,
                "application_info": {
                    "model": self.config.OLLAMA_MODEL,
                    "embedding_model": self.config.EMBEDDING_MODEL,
                    "vector_db_type": self.config.VECTOR_DB_TYPE
                },
                "stats_timestamp": str(pd.Timestamp.now())
            }
        except Exception as e:
            self.logger.error(f"Error getting corpus stats: {e}")
            return {"error": str(e)}
    
    def get_extraction_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of extraction results"""
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        
        # Calculate validation statistics
        validation_errors = 0
        validation_warnings = 0
        for result in successful:
            if 'validation_results' in result:
                validation_results = result['validation_results']
                validation_errors += len(validation_results.get('errors', []))
                validation_warnings += len(validation_results.get('warnings', []))
        
        summary = {
            'total_processed': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'validation_errors': validation_errors,
            'validation_warnings': validation_warnings,
            'successful_invoices': [r.get('invoice_number') for r in successful],
            'failed_files': [r.get('file_path') for r in failed],
            'summary_timestamp': str(pd.Timestamp.now())
        }
        
        return summary
    
    def clear_vector_database(self) -> bool:
        """Clear all documents from vector database"""
        try:
            success = self.rag_pipeline.vector_store_manager.clear_collection()
            if success:
                self.logger.info("Vector database cleared successfully")
            return success
        except Exception as e:
            self.logger.error(f"Error clearing vector database: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform application health check"""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": str(pd.Timestamp.now())
        }
        
        # Check Ollama connection
        try:
            test_response = self.rag_pipeline.llm("Test")
            health["components"]["ollama"] = {"status": "healthy", "model": self.config.OLLAMA_MODEL}
        except Exception as e:
            health["components"]["ollama"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check vector database
        try:
            stats = self.rag_pipeline.vector_store_manager.get_collection_stats()
            health["components"]["vector_db"] = {"status": "healthy", "total_chunks": stats.get("total_chunks", 0)}
        except Exception as e:
            health["components"]["vector_db"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check file system
        try:
            for directory in [self.config.INPUT_DIR, self.config.OUTPUT_DIR]:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
            health["components"]["filesystem"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["filesystem"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "unhealthy"
        
        return health

if __name__ == "__main__":
    app = FinancialRAGApp()
    
    print("🔥 Financial RAG Application initialized!")
    print("📁 Place invoices in: input/invoices/")
    print("📊 Outputs will be saved in: output/")
    print("🧠 Vector database will be stored in: output/vector_db/")
    
    # Perform health check
    health = app.health_check()
    print(f"\n🏥 System Health: {health['status'].upper()}")
