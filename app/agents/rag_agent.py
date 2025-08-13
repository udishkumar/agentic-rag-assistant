# app/agents/rag_agent.py
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import anthropic
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings
import json
from datetime import datetime
import logging
import time
import hashlib
import requests
import torch
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# --- Safe defaults for backends (helps on Apple Silicon / fresh Torch installs) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # disable CUDA if present
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # allow CPU fallback on Mac
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # quieter logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structure for agent responses"""
    answer: str
    sources: List[str]
    confidence: float
    tool_used: str
    metadata: Dict[str, Any]


@dataclass
class Conversation:
    """Structure for a conversation session"""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ConversationManager:
    """Manages persistent conversation storage and retrieval"""
    
    def __init__(self, storage_path: str = "./data/conversations"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.index_file = os.path.join(storage_path, "conversations_index.json")
        self.conversations_index = self._load_index()
        
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load conversation index from disk"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading conversation index: {e}")
        return {}
    
    def _save_index(self):
        """Save conversation index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.conversations_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation index: {e}")
    
    def create_conversation(self, title: str = None) -> Conversation:
        """Create a new conversation"""
        conv_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        if not title:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conversation = Conversation(
            id=conv_id,
            title=title,
            created_at=now,
            updated_at=now,
            messages=[],
            metadata={}
        )
        
        # Update index
        self.conversations_index[conv_id] = {
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0
        }
        
        self._save_index()
        self.save_conversation(conversation)
        
        return conversation
    
    def save_conversation(self, conversation: Conversation):
        """Save a conversation to disk"""
        conv_file = os.path.join(self.storage_path, f"{conversation.id}.json")
        try:
            with open(conv_file, 'w') as f:
                json.dump(asdict(conversation), f, indent=2)
            
            # Update index
            self.conversations_index[conversation.id] = {
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "message_count": len(conversation.messages)
            }
            self._save_index()
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.id}: {e}")
    
    def load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a conversation from disk"""
        conv_file = os.path.join(self.storage_path, f"{conv_id}.json")
        if os.path.exists(conv_file):
            try:
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                return Conversation(**data)
            except Exception as e:
                logger.error(f"Error loading conversation {conv_id}: {e}")
        return None
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation"""
        conv_file = os.path.join(self.storage_path, f"{conv_id}.json")
        try:
            if os.path.exists(conv_file):
                os.remove(conv_file)
            if conv_id in self.conversations_index:
                del self.conversations_index[conv_id]
            self._save_index()
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation {conv_id}: {e}")
            return False
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations sorted by updated_at"""
        conversations = []
        for conv_id, info in self.conversations_index.items():
            conv_info = info.copy()
            conv_info['id'] = conv_id
            conversations.append(conv_info)
        
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return conversations
    
    def update_conversation_title(self, conv_id: str, new_title: str):
        """Update conversation title"""
        conversation = self.load_conversation(conv_id)
        if conversation:
            conversation.title = new_title
            conversation.updated_at = datetime.now().isoformat()
            self.save_conversation(conversation)


class AgenticRAG:
    """Enhanced RAG system with agentic routing and persistent storage"""

    def __init__(
        self,
        anthropic_api_key: str,
        vector_store_path: str = "./data/vector_store",
        upload_path: str = "./data/uploads",
        conversations_path: str = "./data/conversations",
    ):
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.api_key = anthropic_api_key

        self.vector_store_path = vector_store_path
        self.upload_path = upload_path

        # Create directories if they don't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.upload_path, exist_ok=True)

        # Initialize conversation manager
        self.conversation_manager = ConversationManager(conversations_path)
        self.current_conversation: Optional[Conversation] = None

        # Initialize embedding model with optimizations
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            embed_batch_size=128,  # Increased batch size for faster processing
            max_length=256,  # Limit max length for faster processing
            model_kwargs={
                "low_cpu_mem_usage": False,
                "device_map": None,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            },
        )

        # Configure LlamaIndex with optimized settings
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = None
        LlamaSettings.chunk_size = 1024  # Increased chunk size for fewer chunks
        LlamaSettings.chunk_overlap = 128  # Reasonable overlap

        # Initialize ChromaDB for persistent vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=vector_store_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            ),
        )

        # Initialize or load collection
        self._initialize_collection()

        # Document metadata
        self.document_metadata: Dict[str, Any] = {}

        # Index + retrieval
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[VectorIndexRetriever] = None
        self.postprocessor: Optional[SimilarityPostprocessor] = None
        self.storage_context: Optional[StorageContext] = None

        # Track indexed files to prevent re-indexing
        self.indexed_files = set()
        self.metadata_file = os.path.join(self.vector_store_path, "metadata.json")

        # ---- IMPROVED Routing knobs ----
        self.always_enrich_modifications = True
        self.always_enrich_insufficient = True
        self.enrich_low_relevance_threshold = 0.35
        self.always_enrich_places = True
        self.always_enrich_open_ended = True
        self.enrich_length_threshold = 600

        # Load existing metadata
        self._load_metadata()

        # Web search (Google CSE) with simple rate limit / backoff
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self.last_search_time = 0
        self.search_delay = 1.5

        # Try to load existing index
        self._load_existing_index()

        logger.info("AgenticRAG initialized successfully with persistent storage")

    def set_current_conversation(self, conversation: Conversation):
        """Set the current active conversation"""
        self.current_conversation = conversation
        logger.info(f"Switched to conversation: {conversation.id}")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history"""
        if self.current_conversation:
            return self.current_conversation.messages
        return []

    def _initialize_collection(self):
        """Initialize or get existing ChromaDB collection"""
        try:
            self.collection = self.chroma_client.get_collection(name="documents")
            logger.info("Using existing ChromaDB collection")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name="documents", metadata={"description": "Document embeddings for RAG"}
            )
            logger.info("Created new ChromaDB collection")

    def _load_metadata(self):
        """Load metadata from disk"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    self.document_metadata = data.get("documents", {})
                    self.indexed_files = set(data.get("indexed_files", []))
                    logger.info(
                        f"Loaded metadata for {len(self.indexed_files)} indexed files"
                    )
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self.document_metadata = {}
                self.indexed_files = set()

    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            data = {
                "documents": self.document_metadata,
                "indexed_files": list(self.indexed_files),
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def _load_existing_index(self) -> bool:
        """Load existing index from ChromaDB if available"""
        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"Found existing collection with {count} vectors")

                vector_store = ChromaVectorStore(chroma_collection=self.collection)
                self.storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store, embed_model=self.embed_model
                )

                self._create_retriever()

                logger.info(
                    f"Successfully loaded existing index with {len(self.indexed_files)} files"
                )
                return True
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
        return False

    def _create_retriever(self):
        """Create or recreate the retriever (no LLM needed)"""
        if self.index is not None:
            self.retriever = self.index.as_retriever(similarity_top_k=8)
            self.postprocessor = SimilarityPostprocessor(similarity_cutoff=0.2)
            logger.info("Retriever created successfully")

    def load_documents(self, force_reload: bool = False, progress_callback=None) -> bool:
        """Load and index PDF documents with improved performance"""
        try:
            pdf_files = [f for f in os.listdir(self.upload_path) if f.endswith(".pdf")]

            if not pdf_files:
                logger.warning("No PDF files found in upload directory")
                return False

            # Determine which files need indexing
            new_files = []
            for pdf_file in pdf_files:
                file_path = os.path.join(self.upload_path, pdf_file)
                file_hash = self._get_file_hash(file_path)

                if force_reload or file_hash not in self.indexed_files:
                    new_files.append(pdf_file)
                    logger.info(f"Will index: {pdf_file}")
                else:
                    logger.info(f"Already indexed: {pdf_file}")

            if not new_files and not force_reload:
                logger.info("All documents already indexed")
                if self.retriever is None and self.index is not None:
                    self._create_retriever()
                return True

            logger.info(f"Processing {len(new_files)} new PDF files")

            # Load documents with progress tracking
            if new_files:
                if progress_callback:
                    progress_callback(0, f"Loading {len(new_files)} documents...")
                    
                reader = SimpleDirectoryReader(
                    input_files=[os.path.join(self.upload_path, f) for f in new_files],
                    filename_as_id=True,
                )
                new_documents = reader.load_data()
                
                if not new_documents:
                    logger.warning("No new documents could be loaded")
                    return False
                    
                logger.info(f"Loaded {len(new_documents)} new documents")
                
                if progress_callback:
                    progress_callback(20, f"Loaded {len(new_documents)} documents")
            else:
                new_documents = []

            # Create or update index with batch processing
            if self.index is None or (force_reload and pdf_files):
                logger.info("Creating new index with ChromaDB backend...")

                if force_reload:
                    try:
                        self.chroma_client.delete_collection(name="documents")
                        self.collection = self.chroma_client.create_collection(
                            name="documents",
                            metadata={"description": "Document embeddings for RAG"},
                        )
                        self.indexed_files.clear()
                        self.document_metadata.clear()
                        logger.info("Cleared existing collection for force reload")
                    except Exception as e:
                        logger.warning(f"Error clearing collection: {e}")

                if force_reload:
                    reader = SimpleDirectoryReader(
                        self.upload_path, filename_as_id=True, required_exts=[".pdf"]
                    )
                    all_documents = reader.load_data()
                else:
                    all_documents = new_documents

                if progress_callback:
                    progress_callback(30, "Creating vector index...")

                vector_store = ChromaVectorStore(chroma_collection=self.collection)
                self.storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

                # Use batch processing for faster indexing
                self.index = VectorStoreIndex.from_documents(
                    all_documents,
                    storage_context=self.storage_context,
                    embed_model=self.embed_model,
                    show_progress=True,
                    insert_batch_size=32,  # Process in batches
                )

                if progress_callback:
                    progress_callback(80, "Finalizing index...")

                for doc in all_documents:
                    md = getattr(doc, "metadata", {}) or {}
                    path = (
                        md.get("file_path")
                        or md.get("filename")
                        or md.get("file_name")
                        or ""
                    )
                    if path and not os.path.isabs(path):
                        file_path = os.path.join(self.upload_path, os.path.basename(path))
                    else:
                        file_path = path
                    file_name = (
                        os.path.basename(file_path)
                        if file_path
                        else getattr(doc, "doc_id", "unknown")
                    )
                    file_hash = (
                        self._get_file_hash(file_path)
                        if file_path
                        else hashlib.md5(file_name.encode()).hexdigest()
                    )

                    self.document_metadata[file_name] = {
                        "loaded_at": datetime.now().isoformat(),
                        "size": len(getattr(doc, "text", "") or ""),
                        "hash": file_hash,
                    }
                    self.indexed_files.add(file_hash)

            elif new_documents:
                logger.info("Adding new documents to existing index...")
                
                if progress_callback:
                    progress_callback(30, f"Adding {len(new_documents)} documents to index...")
                
                # Batch insert for better performance
                batch_size = 32
                for i in range(0, len(new_documents), batch_size):
                    batch = new_documents[i:i+batch_size]
                    for doc in batch:
                        self.index.insert(doc)
                    
                    if progress_callback:
                        progress = 30 + int((i / len(new_documents)) * 50)
                        progress_callback(progress, f"Indexed {min(i+batch_size, len(new_documents))}/{len(new_documents)} documents")

                for doc in new_documents:
                    md = getattr(doc, "metadata", {}) or {}
                    path = (
                        md.get("file_path")
                        or md.get("filename")
                        or md.get("file_name")
                        or ""
                    )
                    if path and not os.path.isabs(path):
                        file_path = os.path.join(self.upload_path, os.path.basename(path))
                    else:
                        file_path = path
                    file_name = (
                        os.path.basename(file_path)
                        if file_path
                        else getattr(doc, "doc_id", "unknown")
                    )
                    file_hash = (
                        self._get_file_hash(file_path)
                        if file_path
                        else hashlib.md5(file_name.encode()).hexdigest()
                    )
                    self.document_metadata[file_name] = {
                        "loaded_at": datetime.now().isoformat(),
                        "size": len(getattr(doc, "text", "") or ""),
                        "hash": file_hash,
                    }
                    self.indexed_files.add(file_hash)

            # Save metadata + create retriever
            if progress_callback:
                progress_callback(90, "Saving metadata...")
                
            self._save_metadata()
            self._create_retriever()

            if progress_callback:
                progress_callback(100, f"Successfully indexed {len(self.indexed_files)} documents")

            logger.info(f"Successfully indexed {len(self.indexed_files)} total documents")
            return True

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for deduplication"""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()

    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform web search using Google Programmable Search (CSE)."""
        try:
            if not self.google_api_key or not self.google_cse_id:
                logger.warning("Google API key / CSE ID missing – web search disabled.")
                return []

            # Respect a minimal delay between searches
            now = time.time()
            since = now - self.last_search_time
            if since < self.search_delay:
                time.sleep(self.search_delay - since)

            logger.info(f"Performing Google CSE web search for: {query}")
            num = min(max_results, 10)
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": num,
                "safe": "active",
                "gl": "us",
            }

            # Exponential backoff on transient errors
            max_retries = 3
            backoff = 1.5
            for attempt in range(max_retries):
                resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", []) or []
                    self.last_search_time = time.time()

                    results: List[Dict[str, str]] = []
                    for item in items:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", ""),
                        })

                    logger.info(f"Found {len(results)} web search results")
                    return results

                # Handle rate/temporary errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = (attempt + 1) * backoff
                    logger.warning(f"Google CSE transient error {resp.status_code}; retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    continue

                # Non-retryable errors
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                logger.error(f"Google CSE error {resp.status_code}: {err}")
                return []

            return []

        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    def query_documents_with_relevance(
        self, query: str
    ) -> Tuple[Optional[str], float, List[str]]:
        """Query documents and return combined text, avg relevance, and sources"""
        if self.retriever is None:
            logger.warning("Retriever not initialized - no documents indexed")
            return None, 0.0, []

        try:
            logger.info(f"Querying documents for: {query}")
            nodes = self.retriever.retrieve(query)
            if self.postprocessor:
                nodes = self.postprocessor.postprocess_nodes(nodes)

            if not nodes:
                logger.warning("No relevant information found in documents")
                return None, 0.0, []

            scores = [n.score for n in nodes if n.score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            def _content(n):
                if hasattr(n, "get_content") and callable(getattr(n, "get_content")):
                    return n.get_content() or ""
                return getattr(n.node, "text", "") or ""

            kept = [n for n in nodes if (n.score is None or n.score >= 0.2)]
            combined_text = "\n\n".join([_content(n) for n in kept if _content(n).strip()])

            srcs: List[str] = []
            for n in kept:
                md = getattr(n.node, "metadata", {}) or {}
                path = md.get("file_path") or md.get("filename") or md.get("file_name") or ""
                name = os.path.basename(path) if path else (getattr(n.node, "doc_id", "document"))
                page = md.get("page_label") or md.get("page") or md.get("page_number")
                if page is not None:
                    srcs.append(f"{name} (p.{page})")
                else:
                    srcs.append(name)
            
            seen = set()
            srcs = [s for s in srcs if not (s in seen or seen.add(s))]

            if combined_text.strip():
                logger.info(f"Retrieved {len(kept)} relevant nodes with avg score: {avg_score:.3f}")
                return combined_text, avg_score, srcs

            logger.warning("No sufficiently relevant content found after combining")
            return None, 0.0, []

        except Exception as e:
            logger.error(f"Document query error: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, 0.0, []

    def is_response_insufficient(self, response: str) -> bool:
        """Check if Claude's response indicates insufficient information."""
        if not response:
            return True
        rl = response.lower()
        
        kw_hits = any(
            k in rl
            for k in [
                "not provided in", "not mentioned in", "not available in",
                "no information about", "cannot find", "unable to answer based on",
                "outside the scope", "not explicitly stated", "no relevant information",
                "missing", "unavailable", "unspecified",
                "context doesn't provide", "doesn't provide specific",
                "the context doesn't", "document doesn't",
                "you would need to", "need to consult", "need to make",
                "significant modifications", "significantly alter",
                "doesn't contain", "don't have access",
                "apologize", "unable to provide",
                "beyond what's provided", "beyond the",
                "context doesn't have", "not part of",
                "typically not", "generally not",
            ]
        )
        
        re_hits = bool(
            re.search(r"not\s+(provided|mentioned|available|contained)\s+in\s+the\s+(context|text|document|given)", rl)
            or re.search(r"(context|document|text)\s+(doesn't|does not|didn't)\s+(provide|contain|have|include)", rl)
            or re.search(r"information\s+(is|was)\s+(missing|unavailable|not\s+provided)", rl)
            or re.search(r"(would|will)\s+need\s+to\s+(consult|check|search|look)", rl)
            or re.search(r"beyond\s+(what's|what is|the)\s+(provided|given|available)", rl)
        )
        
        return kw_hits or re_hits

    def _is_modification_query(self, text: str) -> bool:
        """Detect queries asking for modifications or variations."""
        if not text:
            return False
        t = text.lower()
        
        # Get conversation history from current conversation
        conversation_history = self.get_conversation_history()
        
        if conversation_history:
            modification_phrases = [
                "make it", "make this", "make that",
                "change it", "modify it", "adjust it",
                "how can i", "how do i", "how to",
                "increase", "decrease", "add more", "reduce",
                "substitute", "replace", "instead of",
                "alternative", "variation", "variant",
                "different way", "other way",
                "keto", "vegan", "vegetarian", "gluten-free", "dairy-free",
                "healthier", "lighter", "richer",
                "more protein", "less carbs", "low fat",
                "it's", "its"
            ]
            
            for phrase in modification_phrases:
                if phrase in t:
                    return True
                    
        return False

    def _is_doc_only_intent(self, text: str) -> bool:
        """Detect when the user explicitly wants answers ONLY from the uploaded docs."""
        if not text:
            return False
        t = text.lower()
        phrases = [
            "according to the document", "from the document", "from the pdf",
            "in the pdf", "based on the context", "use only the document",
            "document only", "pdf only", "strictly from the document",
            "within the document"
        ]
        return any(p in t for p in phrases)

    def _is_open_ended(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(p in t for p in ["tell me about", "overview", "what is", "who is", "where is", "about ", "suggest", "recommend"])

    def _is_place_query(self, text: str) -> bool:
        """Return True if the query looks like a place/attraction (tourism) ask."""
        if not text:
            return False
        t = text.lower().replace("centre", "center")
        place_terms = [
            "museum", "science center", "planetarium", "aquarium", "zoo",
            "observatory", "gallery", "exhibition center", "heritage center",
            "landmark", "monument", "memorial",
            "park", "theme park", "amusement park", "water park", "garden", "arboretum",
            "stadium", "arena", "theater", "theatre",
            "market", "night market", "floating market", "shopping mall", "mall",
            "temple", "shrine", "church", "cathedral", "mosque", "palace", "fort", "castle",
            "old town", "historic center", "historic centre",
            "city hall",
            "science centre for education", "science center for education",
        ]
        if any(term in t for term in place_terms):
            return True
        hints = ["admission", "ticket", "tickets", "opening hours", "hours", "entry fee", "entrance fee", "contact", "website"]
        return any(h in t for h in hints)

    def _is_web_only_intent(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        phrases = [
            "search the web", "search online", "google this", "google it",
            "web only", "use web search", "check the internet", "look up online",
            "search for", "find online"
        ]
        return any(p in t for p in phrases)

    def _is_vague_followup(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(p in t for p in ["variants", "other recipes", "more such", "similar ones", "more", "suggest", "alternatives"])

    def _should_enrich_with_web(self, q: str, answer: str, doc_relevance: float = 1.0) -> bool:
        """Final decision for enrichment with more aggressive triggers."""
        if self._is_doc_only_intent(q):
            return False
            
        if doc_relevance < self.enrich_low_relevance_threshold:
            logger.info(f"Enriching due to low relevance: {doc_relevance:.3f}")
            return True
            
        if self.always_enrich_modifications and self._is_modification_query(q):
            logger.info("Enriching modification query")
            return True
            
        if self.always_enrich_places and self._is_place_query(q):
            logger.info("Enriching place query")
            return True
            
        if self.always_enrich_insufficient and self.is_response_insufficient(answer):
            logger.info("Enriching due to insufficient response")
            return True
            
        if self.always_enrich_open_ended and self._is_open_ended(q):
            logger.info("Enriching open-ended query")
            return True
            
        if len(answer or "") < self.enrich_length_threshold:
            logger.info(f"Enriching due to short response: {len(answer)} chars")
            return True
            
        generic_indicators = [
            "for more", "consult", "refer to", "check with",
            "you could", "you might", "you may want to",
            "it's important to", "remember to", "make sure to",
            "typically", "generally", "usually", "often"
        ]
        if any(ind in answer.lower() for ind in generic_indicators):
            logger.info("Enriching due to generic/padding language")
            return True
            
        return False

    def generate_response(self, query: str, context: str, tool_used: str) -> str:
        """Generate response using Claude API"""
        conversation_context = ""
        conversation_history = self.get_conversation_history()
        
        if conversation_history:
            recent_history = conversation_history[-3:]
            for h in recent_history:
                if 'user' in h and 'assistant' in h:
                    conversation_context += f"User: {h['user']}\nAssistant: {h['assistant']}\n\n"

        if tool_used == "rag_search":
            system_prompt = (
                "You are an intelligent assistant that answers questions based on provided document context. "
                "Use the context verbatim when possible. If the context doesn't fully answer, say what's missing. "
                "Be honest about limitations - if information is incomplete, say so clearly."
            )
            user_prompt = (
                f"Based on the following context from the documents, answer this question: {query}\n\n"
                f"Document Context:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        elif tool_used == "web_search":
            system_prompt = (
                "You are an intelligent assistant that answers questions using web search results.\n"
                "You HAVE been provided search results; do not claim you cannot search the web.\n"
                "If results are sparse, say what's missing, and still give best-effort guidance."
            )
            user_prompt = (
                f"Based on the following web search results, answer this question: {query}\n\n"
                f"Search Results:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        elif tool_used == "combined":
            system_prompt = (
                "You are an intelligent assistant that combines information from documents and web search.\n"
                "You HAVE been provided both; do not claim you cannot search the web.\n"
                "Synthesize information from both sources to provide the most complete answer.\n"
                "Clearly indicate which information comes from documents vs web when appropriate."
            )
            user_prompt = (
                f"Answer this question using both document context and web search results: {query}\n\n"
                f"Available Information:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        else:
            system_prompt = "You are an intelligent, helpful assistant with broad knowledge."
            user_prompt = (
                f"Answer this question to the best of your ability: {query}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )

        try:
            logger.info(f"Generating response with Claude API (tool: {tool_used})")
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2048,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if hasattr(response, "content"):
                if isinstance(response.content, list) and len(response.content) > 0:
                    return response.content[0].text
                elif isinstance(response.content, str):
                    return response.content

            return str(response)

        except anthropic.NotFoundError as e:
            logger.error(f"Claude model not found: {str(e)}")
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2048,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                if hasattr(response, "content"):
                    if isinstance(response.content, list) and len(response.content) > 0:
                        return response.content[0].text
                return str(response)
            except Exception as e2:
                logger.error(f"Fallback model also failed: {str(e2)}")
                return (
                    "I encountered an error generating a response. "
                    "Please check your API key and try again."
                )

        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return f"I encountered an error generating a response: {str(e)}"

    def process_query(self, query: str) -> AgentResponse:
        """Agentic processing pipeline with improved enrichment logic"""
        logger.info(f"Processing query: {query}")

        # Get conversation history from current conversation
        conversation_history = self.get_conversation_history()

        # Expand vague follow-ups using prior user turn
        effective_query = query
        if self._is_vague_followup(query) and conversation_history:
            prev_users = [h.get("user", "") for h in conversation_history if "user" in h]
            if prev_users:
                prev = prev_users[-1]
                if prev and prev.strip():
                    effective_query = f"{query} (context: {prev})"

        all_contexts: List[str] = []
        all_sources: List[str] = []
        tools_used: List[str] = []

        # Web-only flag (skip docs if user asked explicitly)
        web_only = self._is_web_only_intent(query)

        # Step 1: Try document search if available and not web-only
        doc_context = None
        doc_relevance = 0.0
        doc_sources: List[str] = []
        force_web = False

        if not web_only and self.index is not None and self.retriever is not None:
            logger.info("Attempting document search...")
            doc_context, doc_relevance, doc_sources = self.query_documents_with_relevance(effective_query)

            if doc_context:
                initial_response = self.generate_response(query, doc_context, "rag_search")

                if self._should_enrich_with_web(query, initial_response, doc_relevance):
                    logger.info(f"Enrichment triggered (relevance: {doc_relevance:.3f})")
                    all_contexts.append(f"Document Context:\n{doc_context}")
                    all_sources.extend(doc_sources)
                    tools_used.append("rag_search")
                    force_web = True
                else:
                    query_lower = query.lower()
                    needs_current_info = any(
                        word in query_lower
                        for word in ["latest", "current", "recent", "today", "news", "2024", "2025", "now", "update", "new"]
                    )
                    if needs_current_info:
                        logger.info("Current info requested; enriching with web.")
                        all_contexts.append(f"Document Context:\n{doc_context}")
                        all_sources.extend(doc_sources)
                        tools_used.append("rag_search")
                        force_web = True
                    else:
                        if doc_relevance > 0.5 and not self.is_response_insufficient(initial_response):
                            logger.info("Document search provided sufficient answer (no enrichment).")
                            
                            # Save to conversation
                            if self.current_conversation:
                                self.current_conversation.messages.append({
                                    "user": query,
                                    "assistant": initial_response,
                                    "timestamp": datetime.now().isoformat(),
                                    "tool_used": "rag_search",
                                    "sources": doc_sources,
                                    "confidence": max(doc_relevance, 0.7)
                                })
                                self.current_conversation.updated_at = datetime.now().isoformat()
                                self.conversation_manager.save_conversation(self.current_conversation)
                            
                            return AgentResponse(
                                answer=initial_response,
                                sources=doc_sources,
                                confidence=max(doc_relevance, 0.7),
                                tool_used="rag_search",
                                metadata={
                                    "timestamp": datetime.now().isoformat(),
                                    "query_length": len(query),
                                    "response_length": len(initial_response),
                                },
                            )
                        else:
                            logger.info(f"Low confidence ({doc_relevance:.3f}) or insufficient response - forcing web enrichment")
                            all_contexts.append(f"Document Context:\n{doc_context}")
                            all_sources.extend(doc_sources)
                            tools_used.append("rag_search")
                            force_web = True
            else:
                logger.info("No relevant document content found, will search web")
                force_web = True
        else:
            if web_only:
                logger.info("User requested web-only search; skipping document retrieval.")
            else:
                logger.info("No documents indexed, will use web search or general knowledge")
            force_web = True

        # Step 2: Perform web search when needed/forced
        query_lower = query.lower()
        needs_web_search = (
            web_only
            or force_web
            or any(
                word in query_lower
                for word in ["search", "find", "latest", "current", "recent", "news", "today", "2024", "2025", "update", "new", "google"]
            )
            or (self.index is None and any(word in query_lower for word in ["what", "when", "where", "who", "how", "why"]))
        )

        if needs_web_search:
            logger.info("Performing web search for comprehensive answer...")

            search_results = self.web_search(effective_query)
            if search_results:
                web_context = json.dumps(search_results, indent=2)
                all_contexts.append(f"Web Search Results:\n{web_context}")
                all_sources.extend([r["link"] for r in search_results[:3] if r.get("link")])
                tools_used.append("web_search")

        # Step 3: Generate final response
        if all_contexts:
            combined_context = "\n\n".join(all_contexts)
            tool_description = "combined" if len(tools_used) > 1 else tools_used[0]
            final_response = self.generate_response(query, combined_context, tool_description)

            if len(tools_used) > 1:
                confidence = 0.9
            elif "rag_search" in tools_used:
                confidence = max(doc_relevance, 0.7)
            else:
                confidence = 0.75

            logger.info(f"Generated response using: {', '.join(tools_used)}")

        else:
            logger.info("Using general knowledge")
            final_response = self.generate_response(query, "", "general")
            confidence = 0.6
            tool_description = "general_knowledge"

        # Adjust confidence if answer still signals insufficiency
        if self.is_response_insufficient(final_response):
            confidence = min(confidence, 0.55)

        # Store in conversation
        if self.current_conversation:
            self.current_conversation.messages.append({
                "user": query,
                "assistant": final_response,
                "timestamp": datetime.now().isoformat(),
                "tool_used": " + ".join(tools_used) if tools_used else tool_description,
                "sources": all_sources,
                "confidence": confidence
            })
            self.current_conversation.updated_at = datetime.now().isoformat()
            self.conversation_manager.save_conversation(self.current_conversation)

        return AgentResponse(
            answer=final_response,
            sources=all_sources,
            confidence=confidence,
            tool_used=" + ".join(tools_used) if tools_used else tool_description,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query),
                "response_length": len(final_response),
                "tools_used": tools_used,
                "doc_relevance": doc_relevance if (doc_context is not None) else 0,
            },
        )

    def clear_conversation(self):
        """Clear current conversation"""
        if self.current_conversation:
            self.current_conversation.messages = []
            self.current_conversation.updated_at = datetime.now().isoformat()
            self.conversation_manager.save_conversation(self.current_conversation)
        logger.info("Conversation history cleared")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        conversation_count = len(self.get_conversation_history()) if self.current_conversation else 0
        
        return {
            "documents_loaded": len(self.document_metadata),
            "document_list": list(self.document_metadata.keys()),
            "conversation_length": conversation_count,
            "index_ready": self.index is not None and self.retriever is not None,
            "vector_store": "ChromaDB (Persistent)",
            "last_query": self.current_conversation.messages[-1] if self.current_conversation and self.current_conversation.messages else None,
            "indexed_files_count": len(self.indexed_files),
            "collection_count": self.collection.count() if self.collection else 0,
            "current_conversation_id": self.current_conversation.id if self.current_conversation else None,
        }