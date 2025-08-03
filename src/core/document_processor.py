import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self):
        from ..config import Config
        self.config = Config()
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']
        self.logger = logging.getLogger(__name__)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single document and return Document objects"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._load_pdf(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff']:
            return self._load_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF using LangChain PyPDFLoader"""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Add enhanced metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'file_type': 'pdf',
                    'page_number': i + 1,
                    'total_pages': len(documents),
                    'file_size': file_path.stat().st_size,
                    'document_id': f"{file_path.stem}_{i}"
                })
            
            self.logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading PDF {file_path}: {e}")
            # Return empty document with error info
            return [Document(
                page_content="",
                metadata={
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'error': str(e),
                    'file_type': 'pdf'
                }
            )]
    
    def _load_image(self, file_path: Path) -> List[Document]:
        """Load image using OCR (fallback to pytesseract if needed)"""
        try:
            # Try using LangChain's image capabilities first
            # If not available, fall back to pytesseract
            try:
                from langchain_community.document_loaders import UnstructuredImageLoader
                loader = UnstructuredImageLoader(str(file_path))
                documents = loader.load()
            except ImportError:
                # Fallback to pytesseract
                import pytesseract
                from PIL import Image
                
                image = Image.open(file_path)
                # Preprocess image for better OCR
                image = self._preprocess_image(image)
                text = pytesseract.image_to_string(image)
                
                documents = [Document(
                    page_content=text,
                    metadata={}
                )]
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'file_type': 'image',
                    'file_size': file_path.stat().st_size,
                    'document_id': file_path.stem
                })
            
            self.logger.info(f"Loaded image document from {file_path.name}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading image {file_path}: {e}")
            return [Document(
                page_content="",
                metadata={
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'error': str(e),
                    'file_type': 'image'
                }
            )]
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        try:
            from PIL import Image
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize if too small
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def load_all_documents(self, folder_path: Union[str, Path]) -> List[Document]:
        """Load all supported documents from a folder"""
        folder_path = Path(folder_path)
        all_documents = []
        
        if not folder_path.exists():
            self.logger.error(f"Folder does not exist: {folder_path}")
            return all_documents
        
        # Get all supported files
        supported_files = []
        for ext in self.supported_formats:
            supported_files.extend(folder_path.glob(f"*{ext}"))
        
        if not supported_files:
            self.logger.warning(f"No supported files found in {folder_path}")
            return all_documents
        
        self.logger.info(f"Processing {len(supported_files)} files...")
        
        for file_path in supported_files:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        self.logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def extract_text(self, file_path: Union[str, Path]) -> str:
        """Extract plain text from document (backward compatibility)"""
        documents = self.load_document(file_path)
        
        if not documents:
            return ""
        
        # Combine all pages/documents into single text
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        return combined_text
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['chunk_id'] = f"{chunk.metadata.get('document_id', 'unknown')}_{i}"
            
            self.logger.info(f"Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting documents: {e}")
            return documents  # Return original documents if splitting fails
    
    def validate_document(self, file_path: Union[str, Path]) -> bool:
        """Validate if document format is supported"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def get_document_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get basic information about the document"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_size_formatted': self._format_file_size(file_path.stat().st_size),
            'file_extension': file_path.suffix.lower(),
            'file_type': 'pdf' if file_path.suffix.lower() == '.pdf' else 'image',
            'is_supported': self.validate_document(file_path)
        }
        
        # Add PDF-specific info
        if file_path.suffix.lower() == '.pdf':
            try:
                documents = self._load_pdf(file_path)
                info['page_count'] = len(documents)
            except Exception:
                info['page_count'] = 'Unknown'
        
        return info
    
    def get_folder_stats(self, folder_path: Union[str, Path]) -> Dict[str, Any]:
        """Get statistics about documents in a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return {'error': 'Folder does not exist'}
        
        supported_files = []
        for ext in self.supported_formats:
            supported_files.extend(folder_path.glob(f"*{ext}"))
        
        stats = {
            'total_files': len(supported_files),
            'file_types': {},
            'files': []
        }
        
        for file_path in supported_files:
            file_info = self.get_document_info(file_path)
            stats['files'].append(file_info)
            
            # Count file types
            file_type = file_info['file_type']
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
        
        return stats
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
