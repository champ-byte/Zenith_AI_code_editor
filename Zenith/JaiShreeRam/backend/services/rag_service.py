import os
import time
import shutil
import logging
import gc
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RAGService:
    # Extensions to index
    ALLOWED_EXTS = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".cs", 
        ".go", ".rs", ".php", ".rb", ".html", ".css", ".sql", ".md", ".json"
    }
    
    # Directories to ignore
    IGNORE_DIRS = {"node_modules", ".git", "__pycache__", "venv", ".env", "chroma_db_local", "dist", "build"}

    def __init__(self):
        # Initialize local embeddings
        try:
            # Using a lightweight, high-performance local model
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Initialized local HuggingFace embeddings (all-MiniLM-L6-v2)")
        except Exception as e:
            logger.error(f"Failed to initialize local embeddings: {str(e)}")
            self.embeddings = None

        self.persist_directory = os.path.join(os.getcwd(), "chroma_db_local")
        self.vector_store = None
        self.current_indexed_path = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize or load the vector store"""
        if self.embeddings and os.path.exists(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="codebase_context_local"
                )
                logger.info("Chroma vector store initialized with local embeddings")
            except Exception as e:
                logger.error(f"Failed to initialize Chroma: {str(e)}")

    def _clear_existing_index(self):
        """Robustly clear the existing vector store"""
        # 1. Try using the API if instance exists
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                logger.info("Deleted Chroma collection via API")
            except Exception as e:
                logger.warning(f"Failed to delete collection via API: {e}")
            self.vector_store = None

        # Force garbage collection to release file handles
        gc.collect()

        # 2. Try to force cleanup the directory
        if os.path.exists(self.persist_directory):
            # Windows file locking retry loop
            for attempt in range(3):
                try:
                    if attempt > 0: time.sleep(1)
                    if os.path.exists(self.persist_directory):
                        shutil.rmtree(self.persist_directory)
                    logger.info(f"Removed persistence directory: {self.persist_directory}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/3 to remove persistence directory failed: {e}")

    def index_codebase(self, directory_path: str) -> Dict[str, Any]:
        """Index the codebase from the given directory"""
        if not self.embeddings:
            return {"success": False, "error": "Embeddings not configured (failed to load local model)"}
            
        try:
            if not os.path.exists(directory_path):
                return {"success": False, "error": f"Directory not found: {directory_path}"}

            logger.info(f"Indexing codebase from {directory_path}")

            # CLEAR EXISTING INDEX FIRST
            self._clear_existing_index()
            self.current_indexed_path = directory_path
            
            documents = []
            for root, dirs, files in os.walk(directory_path):
                # Prune ignored directories
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith('.')]
                
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.ALLOWED_EXTS:
                        file_path = os.path.join(root, file)
                        try:
                            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                            docs = loader.load()
                            # Add metadata
                            for doc in docs:
                                doc.metadata["source"] = file_path
                                doc.metadata["filename"] = file
                            documents.extend(docs)
                        except Exception as e:
                            logger.warning(f"Failed to load file {file_path}: {e}")

            if not documents:
                return {"success": False, "error": "No valid documents found to index"}

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )
            splits = text_splitter.split_documents(documents)
            
            logger.info(f"Created {len(splits)} chunks from {len(documents)} files")

            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="codebase_context_local"
            )
            
            return {
                "success": True, 
                "message": f"Indexed {len(documents)} files ({len(splits)} chunks) using local embeddings",
                "chunks": len(splits),
                "files": len(documents)
            }

        except Exception as e:
            logger.error(f"Error indexing codebase: {str(e)}")
            return {"success": False, "error": str(e)}

    def retrieve_context(self, query: str, top_k: int = 120) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if not self.vector_store:
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=top_k)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def reset_index(self):
        """Reset the vector store and current path"""
        try:
            self._clear_existing_index()
            self.current_indexed_path = None
            logger.info("RAG index reset")
            return {"success": True, "message": "RAG index reset"}
        except Exception as e:
            logger.error(f"Error resetting index: {str(e)}")
            return {"success": False, "error": str(e)}

    def _build_file_tree(self, startpath: str) -> str:
        """Build text representation of the file tree"""
        if not startpath: return ""
        lines = [f"Project Root: {os.path.basename(startpath)}"]
        
        for root, dirs, files in os.walk(startpath):
            # Skip hidden and ignored directories
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith('.')]

            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            
            if root != startpath:
                lines.append(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
            else:
                subindent = ' ' * 4
                
            for f in files:
                if not f.startswith('.'):
                    lines.append(f"{subindent}{f}")
                    
        return "\n".join(lines)

    def query_with_context(self, query: str) -> Dict[str, Any]:
        """Get formatted context string for a query"""
        docs = self.retrieve_context(query)
        if not docs:
            return {"context": "", "sources": [], "file_tree": ""}

        # Extract unique filenames
        sources = []
        for doc in docs:
            filename = doc.metadata.get("filename")
            if filename and filename not in sources:
                sources.append(filename)
        sources.sort()

        # Build context string
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("filename", "unknown")
            content = doc.page_content
            context_parts.append(f"File: {source}\nContent:\n{content}\n---")

        return {
            "context": "\n".join(context_parts),
            "sources": sources,
            "file_tree": self._build_file_tree(self.current_indexed_path) if self.current_indexed_path else "No file tree available"
        }
