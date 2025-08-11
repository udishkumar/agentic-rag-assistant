# app/agents/rag_agent.py
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
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
import torch  # <-- for safe CPU dtype and avoiding meta tensors

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


class AgenticRAG:
    """Enhanced RAG system with agentic routing and persistent storage"""

    def __init__(
        self,
        anthropic_api_key: str,
        vector_store_path: str = "./data/vector_store",
        upload_path: str = "./data/uploads",
    ):
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.api_key = anthropic_api_key

        self.vector_store_path = vector_store_path
        self.upload_path = upload_path

        # Create directories if they don't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.upload_path, exist_ok=True)

        # Initialize embedding model (force CPU; avoid meta tensors/device_map sharding)
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            embed_batch_size=32,
            model_kwargs={
                "low_cpu_mem_usage": False,   # disable meta-init path
                "device_map": None,           # no accelerate sharding
                "torch_dtype": torch.float32, # stable on CPU
            },
        )

        # Configure LlamaIndex to NOT use an internal LLM
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = None
        LlamaSettings.chunk_size = 512
        LlamaSettings.chunk_overlap = 50

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

        # Conversation memory + metadata
        self.conversation_history: List[Dict[str, Any]] = []
        self.document_metadata: Dict[str, Any] = {}

        # Index + retrieval
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[VectorIndexRetriever] = None
        self.postprocessor: Optional[SimilarityPostprocessor] = None
        self.storage_context: Optional[StorageContext] = None

        # Track indexed files to prevent re-indexing
        self.indexed_files = set()
        self.metadata_file = os.path.join(self.vector_store_path, "metadata.json")

        # ---- Routing knobs (guarantee enrichment) ----
        self.always_enrich_places = True
        self.always_enrich_open_ended = True
        self.enrich_length_threshold = 450  # if the doc-only answer is shorter than this, enrich

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

    def load_documents(self, force_reload: bool = False) -> bool:
        """Load and index PDF documents with proper persistence"""
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

            # Load documents
            if new_files:
                reader = SimpleDirectoryReader(
                    input_files=[os.path.join(self.upload_path, f) for f in new_files],
                    filename_as_id=True,
                )
                new_documents = reader.load_data()
                if not new_documents:
                    logger.warning("No new documents could be loaded")
                    return False
                logger.info(f"Loaded {len(new_documents)} new documents")
            else:
                new_documents = []

            # Create or update index
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

                vector_store = ChromaVectorStore(chroma_collection=self.collection)
                self.storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

                self.index = VectorStoreIndex.from_documents(
                    all_documents,
                    storage_context=self.storage_context,
                    embed_model=self.embed_model,
                    show_progress=True,
                )

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
                for doc in new_documents:
                    self.index.insert(doc)

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
            self._save_metadata()
            self._create_retriever()

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
            num = min(max_results, 10)  # Google CSE supports up to 10 results per request
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": num,
                "safe": "active",  # optional
                "gl": "us",        # optional (geo bias)
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

            # Keep what survived postprocessing; include if score is None
            def _content(n):
                # LlamaIndex NodeWithScore supports .get_content(); fall back to n.node.text
                if hasattr(n, "get_content") and callable(getattr(n, "get_content")):
                    return n.get_content() or ""
                return getattr(n.node, "text", "") or ""

            kept = [n for n in nodes if (n.score is None or n.score >= 0.2)]
            combined_text = "\n\n".join([_content(n) for n in kept if _content(n).strip()])

            # Build human-friendly sources with page numbers when available
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
            # Deduplicate sources preserving order
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

    # ------------------ Heuristics ------------------

    def is_response_insufficient(self, response: str) -> bool:
        """Check if Claude's response indicates insufficient information (robust)."""
        if not response:
            return True
        rl = response.lower()
        kw_hits = any(
            k in rl
            for k in [
                "not provided in", "not mentioned in", "not available in",
                "no information about", "cannot find", "unable to answer based on",
                "outside the scope", "not explicitly stated", "no relevant information",
                "missing", "unavailable", "unspecified"
            ]
        )
        re_hits = bool(
            re.search(r"not\s+(provided|mentioned|available)\s+in\s+the\s+(context|text|document)", rl)
            or re.search(r"information\s+(is|was)\s+(missing|unavailable|not\s+provided)", rl)
        )
        return kw_hits or re_hits

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
        return any(p in t for p in ["tell me about", "overview", "what is", "who is", "where is", "about "])

    def _is_place_query(self, text: str) -> bool:
        """Return True if the query looks like a place/attraction (tourism) ask."""
        if not text:
            return False
        t = text.lower().replace("centre", "center")
        place_terms = [
            # general attractions
            "museum", "science center", "planetarium", "aquarium", "zoo",
            "observatory", "gallery", "exhibition center", "heritage center",
            "landmark", "monument", "memorial",
            # parks & entertainment
            "park", "theme park", "amusement park", "water park", "garden", "arboretum",
            "stadium", "arena", "theater", "theatre",
            # markets & shopping
            "market", "night market", "floating market", "shopping mall", "mall",
            # religious/historic sites
            "temple", "shrine", "church", "cathedral", "mosque", "palace", "fort", "castle",
            "old town", "historic center", "historic centre",
            # admin/civic
            "city hall",
            # specific variants for your case
            "science centre for education", "science center for education",
        ]
        if any(term in t for term in place_terms):
            return True
        hints = ["admission", "ticket", "tickets", "opening hours", "hours", "entry fee", "entrance fee", "contact", "website"]
        return any(h in t for h in hints)

    def _should_enrich_with_web(self, q: str, answer: str) -> bool:
        """Final decision for enrichment. Guarantees web enrichment for places and open-ended asks,
        unless the user explicitly asked for doc-only answers."""
        if self._is_doc_only_intent(q):
            return False
        if self.always_enrich_places and self._is_place_query(q):
            return True
        if self.is_response_insufficient(answer):
            return True
        if self.always_enrich_open_ended and self._is_open_ended(q):
            return True
        if len(answer or "") < self.enrich_length_threshold:
            return True
        return False

    # ------------------ LLM Orchestration ------------------

    def generate_response(self, query: str, context: str, tool_used: str) -> str:
        """Generate response using Claude API"""
        conversation_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]
            for h in recent_history:
                conversation_context += f"User: {h['user']}\nAssistant: {h['assistant']}\n\n"

        if tool_used == "rag_search":
            system_prompt = (
                "You are an intelligent assistant that answers questions based on provided document context. "
                "Use the context verbatim when possible. If the context doesn't fully answer, say what's missing."
            )
            user_prompt = (
                f"Based on the following context from the documents, answer this question: {query}\n\n"
                f"Document Context:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        elif tool_used == "web_search":
            system_prompt = (
                "You are an intelligent assistant that answers questions using web search results. "
                "Synthesize information from multiple sources and be up-to-date."
            )
            user_prompt = (
                f"Based on the following web search results, answer this question: {query}\n\n"
                f"Search Results:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        elif tool_used == "combined":
            system_prompt = (
                "You are an intelligent assistant that combines information from documents and web search. "
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
        """Agentic processing pipeline: docs → web → general"""
        logger.info(f"Processing query: {query}")

        all_contexts: List[str] = []
        all_sources: List[str] = []
        tools_used: List[str] = []

        # Step 1: Try document search if available
        doc_context = None
        doc_relevance = 0.0
        doc_sources: List[str] = []
        force_web = False

        if self.index is not None and self.retriever is not None:
            logger.info("Attempting document search...")
            doc_context, doc_relevance, doc_sources = self.query_documents_with_relevance(query)

            if doc_context and doc_relevance > 0.2:
                # Generate initial response from documents
                initial_response = self.generate_response(query, doc_context, "rag_search")

                # Decide whether to enrich with web
                if self._should_enrich_with_web(query, initial_response):
                    logger.info("Enrichment triggered: adding web search (doc context retained).")
                    all_contexts.append(f"Document Context:\n{doc_context}")
                    all_sources.extend(doc_sources)
                    tools_used.append("rag_search")
                    force_web = True
                else:
                    # Check if query needs current info
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
                        logger.info("Document search provided sufficient answer (no enrichment).")
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
                logger.info(f"Document search relevance too low ({doc_relevance:.3f}), will search web")
        else:
            logger.info("No documents indexed, will use web search or general knowledge")

        # Step 2: Perform web search when needed/forced
        query_lower = query.lower()
        needs_web_search = (
            force_web
            or any(
                word in query_lower
                for word in ["search", "find", "latest", "current", "recent", "news", "today", "2024", "2025", "update", "new"]
            )
            or (self.index is None and any(word in query_lower for word in ["what", "when", "where", "who", "how", "why"]))
        )

        if needs_web_search:
            logger.info("Performing web search for comprehensive answer...")
            search_results = self.web_search(query)
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

        # Store in conversation history
        self.conversation_history.append(
            {"user": query, "assistant": final_response, "timestamp": datetime.now().isoformat()}
        )
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

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
                "doc_relevance": doc_relevance if doc_context else 0,
            },
        )

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        return {
            "documents_loaded": len(self.document_metadata),
            "document_list": list(self.document_metadata.keys()),
            "conversation_length": len(self.conversation_history),
            "index_ready": self.index is not None and self.retriever is not None,
            "vector_store": "ChromaDB (Persistent)",
            "last_query": self.conversation_history[-1] if self.conversation_history else None,
            "indexed_files_count": len(self.indexed_files),
            "collection_count": self.collection.count() if self.collection else 0,
        }
