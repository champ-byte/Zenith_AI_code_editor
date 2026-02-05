import os
import time
import shutil
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Keep logging simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    # Extensions to index
    ALLOWED_EXTS = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".cs", 
        ".go", ".rs", ".php", ".rb", ".html", ".css", ".sql", ".md", ".json"
    }
    
    # Directories to ignore
    IGNORE_DIRS = {"node_modules", ".git", "__pycache__", "venv", ".env", "chroma_db_local"}

    def __init__(self):
        self.persist_directory = os.path.join(os.getcwd(), "chroma_db_local")
        self.vector_store = None
        self.current_indexed_path = None
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Embeddings init failed: {e}")
            self.embeddings = None

        # Load existing db if available
        if self.embeddings and os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="codebase_context_local"
            )

    def _clear_index(self):
        """Clears the current vector store and deletes the persistence directory."""
        # 1. Try to delete collection via API if possible (cleanest way)
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                logger.info("Collection deleted via API")
            except Exception as e:
                logger.warning(f"API delete_collection failed: {e}")
            
            self.vector_store = None
        
        # Force garbage collection to release file handles
        import gc
        gc.collect()

        # 2. Try to nuke directory (Fallback)
        if os.path.exists(self.persist_directory):
            # Windows file locking retry loop
            for attempt in range(3):
                try:
                    # Windows sometimes holds file locks, so we wait a bit
                    if attempt > 0: time.sleep(1)
                    
                    if os.path.exists(self.persist_directory):
                        shutil.rmtree(self.persist_directory, ignore_errors=True)
                    logger.info("Persistence directory cleared")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/3 to delete folder failed: {e}")
                    # If we can't delete it, we just proceed. Chroma will likely just use the existing folder
                    # which is empty now if delete_collection worked.


    def index_codebase(self, directory_path: str):
        if not self.embeddings:
            return {"success": False, "error": "Embeddings not initialized"}
            
        if not os.path.exists(directory_path):
            return {"success": False, "error": f"Path not found: {directory_path}"}

        try:
            print(f"🔄 [RAG] Starting indexing for: {directory_path}")
            logger.info(f"Indexing {directory_path}...")
            self._clear_index()
            self.current_indexed_path = directory_path
            
            documents = []
            for root, dirs, files in os.walk(directory_path):
                # Prune ignored directories
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith('.')]
                
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.ALLOWED_EXTS:
                        path = os.path.join(root, file)
                        try:
                            loader = TextLoader(path, encoding='utf-8', autodetect_encoding=True)
                            docs = loader.load()
                            for d in docs:
                                d.metadata.update({"source": path, "filename": file})
                            documents.extend(docs)
                        except Exception as e:
                            logger.warning(f"Skipping {file}: {e}")

            if not documents:
                return {"success": False, "error": "No documents found to index"}

            # Split and Store
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="codebase_context_local"
            )
            
            print(f"✅ [RAG] Indexing complete! {len(documents)} files processed.")
            
            return {
                "success": True, 
                "message": f"Indexed {len(documents)} files ({len(splits)} chunks)",
                "chunks": len(splits), 
                "files": len(documents)
            }

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return {"success": False, "error": str(e)}

    def retrieve_context(self, query: str, top_k: int = 120):
        if not self.vector_store:
            return []
        try:
            return self.vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def reset_index(self):
        self._clear_index()
        self.current_indexed_path = None
        return {"success": True, "message": "Index reset"}

    def _build_file_tree(self, startpath):
        if not startpath: return ""
        lines = [f"Project Root: {os.path.basename(startpath)}"]
        
        for root, dirs, files in os.walk(startpath):
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

    def query_with_context(self, query: str):
        docs = self.retrieve_context(query)
    
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
            filename = doc.metadata.get("filename", "unknown")
            content = doc.page_content
            context_parts.append(f"File: {filename}\nContent:\n{content}\n---")
    
        context_str = "\n".join(context_parts)
    
        return {
            "context": context_str,
            "sources": sources,
            "file_tree": self._build_file_tree(self.current_indexed_path)
        }

