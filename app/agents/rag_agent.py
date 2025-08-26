# app/agents/rag_agent.py - Enhanced with True Agentic AI Capabilities
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
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
from datetime import datetime, timedelta
import logging
import time
import hashlib
import requests
import torch
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict, deque
from enum import Enum
import pickle

# --- Safe defaults for backends (helps on Apple Silicon / fresh Torch installs) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types for adaptive routing"""
    DOCUMENT_SPECIFIC = "document_specific"
    WEB_CURRENT = "web_current"
    HYBRID = "hybrid"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MODIFICATION = "modification"
    PLACE_INFO = "place_info"
    OPEN_ENDED = "open_ended"


class GoalState(Enum):
    """States for goal tracking"""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class AgentGoal:
    """Represents a high-level goal the agent is pursuing"""
    id: str
    description: str
    created_at: str
    state: GoalState
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    completion_criteria: List[str] = field(default_factory=list)


@dataclass
class ExperienceMemory:
    """Stores learned patterns and successful strategies"""
    query_pattern: str
    successful_strategy: str
    tool_sequence: List[str]
    confidence_score: float
    success_count: int = 0
    failure_count: int = 0
    last_used: str = ""
    context_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Tracks agent performance for self-improvement"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_confidence: float = 0.0
    avg_response_time: float = 0.0
    tool_effectiveness: Dict[str, float] = field(default_factory=dict)
    query_type_performance: Dict[str, float] = field(default_factory=dict)
    user_satisfaction_score: float = 0.0


@dataclass
class AgentResponse:
    """Enhanced structure for agent responses with learning metadata"""
    answer: str
    sources: List[str]
    confidence: float
    tool_used: str
    metadata: Dict[str, Any]
    reasoning_chain: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    goal_progress: Optional[Dict[str, Any]] = None
    self_evaluation: Optional[Dict[str, Any]] = None


@dataclass
class Conversation:
    """Structure for a conversation session"""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    active_goals: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class AgentLearningSystem:
    """Manages the agent's learning and adaptation capabilities"""
    
    def __init__(self, storage_path: str = "./data/agent_memory"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Experience memory for pattern learning
        self.experiences: List[ExperienceMemory] = []
        self.experience_index: Dict[str, List[int]] = defaultdict(list)
        
        # Performance tracking
        self.performance = PerformanceMetrics()
        
        # User preference learning
        self.user_preferences: Dict[str, Any] = {
            "response_style": "balanced",
            "detail_level": "moderate",
            "preferred_sources": [],
            "avoided_topics": [],
            "interaction_patterns": []
        }
        
        # Query pattern recognition
        self.query_patterns: Dict[str, int] = defaultdict(int)
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            "confidence_threshold": 0.7,
            "relevance_threshold": 0.35,
            "web_enrichment_threshold": 0.6
        }
        
        # Load existing memory
        self.load_memory()
    
    def record_experience(self, query: str, strategy: str, tools: List[str], 
                         confidence: float, success: bool, context: Dict[str, Any]):
        """Record a new experience for learning"""
        pattern = self._extract_pattern(query)
        
        # Check if similar experience exists
        existing_idx = self._find_similar_experience(pattern, strategy)
        
        if existing_idx is not None:
            # Update existing experience
            exp = self.experiences[existing_idx]
            if success:
                exp.success_count += 1
            else:
                exp.failure_count += 1
            exp.confidence_score = self._calculate_confidence(exp)
            exp.last_used = datetime.now().isoformat()
        else:
            # Create new experience
            exp = ExperienceMemory(
                query_pattern=pattern,
                successful_strategy=strategy,
                tool_sequence=tools,
                confidence_score=confidence,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                last_used=datetime.now().isoformat(),
                context_features=context
            )
            self.experiences.append(exp)
            self.experience_index[pattern].append(len(self.experiences) - 1)
        
        # Update performance metrics
        self.performance.total_queries += 1
        if success:
            self.performance.successful_queries += 1
        else:
            self.performance.failed_queries += 1
        
        self.performance.avg_confidence = (
            (self.performance.avg_confidence * (self.performance.total_queries - 1) + confidence) 
            / self.performance.total_queries
        )
        
        # Save memory periodically
        if self.performance.total_queries % 10 == 0:
            self.save_memory()
    
    def suggest_strategy(self, query: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest best strategy based on learned experiences"""
        pattern = self._extract_pattern(query)
        
        # Find relevant experiences
        relevant_experiences = []
        for idx in self.experience_index.get(pattern, []):
            exp = self.experiences[idx]
            if exp.confidence_score > 0.6:
                relevant_experiences.append(exp)
        
        # Sort by confidence and recency
        relevant_experiences.sort(
            key=lambda x: (x.confidence_score, x.last_used), 
            reverse=True
        )
        
        if relevant_experiences:
            best_exp = relevant_experiences[0]
            return {
                "strategy": best_exp.successful_strategy,
                "tools": best_exp.tool_sequence,
                "confidence": best_exp.confidence_score,
                "reasoning": f"Based on {best_exp.success_count} successful similar queries"
            }
        
        return None
    
    def adapt_thresholds(self):
        """Adapt decision thresholds based on performance"""
        if self.performance.total_queries > 50:
            # Adjust confidence threshold based on success rate
            success_rate = self.performance.successful_queries / self.performance.total_queries
            
            if success_rate < 0.7:
                # Lower thresholds if struggling
                self.adaptive_thresholds["confidence_threshold"] *= 0.95
                self.adaptive_thresholds["web_enrichment_threshold"] *= 0.9
            elif success_rate > 0.85:
                # Raise thresholds if performing well
                self.adaptive_thresholds["confidence_threshold"] *= 1.02
                self.adaptive_thresholds["web_enrichment_threshold"] *= 1.05
            
            # Keep thresholds in reasonable bounds
            self.adaptive_thresholds["confidence_threshold"] = max(0.5, min(0.9, 
                self.adaptive_thresholds["confidence_threshold"]))
            self.adaptive_thresholds["web_enrichment_threshold"] = max(0.4, min(0.8,
                self.adaptive_thresholds["web_enrichment_threshold"]))
    
    def learn_user_preference(self, feedback_type: str, context: Dict[str, Any]):
        """Learn from user feedback to improve future responses"""
        if feedback_type == "positive":
            # Reinforce current approach
            if "response_length" in context:
                self.user_preferences["detail_level"] = context["response_length"]
            if "sources_used" in context:
                self.user_preferences["preferred_sources"].extend(context["sources_used"])
        elif feedback_type == "negative":
            # Adjust approach
            if "response_length" in context:
                if context["response_length"] == "long":
                    self.user_preferences["detail_level"] = "brief"
                else:
                    self.user_preferences["detail_level"] = "detailed"
    
    def _extract_pattern(self, query: str) -> str:
        """Extract pattern from query for similarity matching"""
        # Simple pattern extraction - can be enhanced with NLP
        query_lower = query.lower()
        
        # Identify key patterns
        patterns = []
        if "what" in query_lower: patterns.append("what_query")
        if "how" in query_lower: patterns.append("how_query")
        if "when" in query_lower: patterns.append("when_query")
        if "where" in query_lower: patterns.append("where_query")
        if "why" in query_lower: patterns.append("why_query")
        
        # Identify domain patterns
        if any(term in query_lower for term in ["code", "program", "function", "algorithm"]):
            patterns.append("technical")
        if any(term in query_lower for term in ["latest", "current", "recent", "news"]):
            patterns.append("current_info")
        if any(term in query_lower for term in ["document", "pdf", "file"]):
            patterns.append("document_specific")
        
        return "_".join(patterns) if patterns else "general"
    
    def _find_similar_experience(self, pattern: str, strategy: str) -> Optional[int]:
        """Find similar experience in memory"""
        for idx in self.experience_index.get(pattern, []):
            exp = self.experiences[idx]
            if exp.successful_strategy == strategy:
                return idx
        return None
    
    def _calculate_confidence(self, exp: ExperienceMemory) -> float:
        """Calculate confidence score for an experience"""
        total = exp.success_count + exp.failure_count
        if total == 0:
            return 0.5
        
        success_rate = exp.success_count / total
        
        # Apply time decay
        last_used = datetime.fromisoformat(exp.last_used)
        days_old = (datetime.now() - last_used).days
        time_factor = max(0.5, 1.0 - (days_old * 0.01))
        
        return success_rate * time_factor
    
    def save_memory(self):
        """Persist learning to disk"""
        memory_data = {
            "experiences": [asdict(exp) for exp in self.experiences],
            "performance": asdict(self.performance),
            "user_preferences": self.user_preferences,
            "adaptive_thresholds": self.adaptive_thresholds,
            "query_patterns": dict(self.query_patterns)
        }
        
        memory_file = os.path.join(self.storage_path, "agent_memory.json")
        try:
            with open(memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
            logger.info("Agent memory saved successfully")
        except Exception as e:
            logger.error(f"Error saving agent memory: {e}")
    
    def load_memory(self):
        """Load learning from disk"""
        memory_file = os.path.join(self.storage_path, "agent_memory.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                
                self.experiences = [ExperienceMemory(**exp) for exp in memory_data.get("experiences", [])]
                self.performance = PerformanceMetrics(**memory_data.get("performance", {}))
                self.user_preferences = memory_data.get("user_preferences", self.user_preferences)
                self.adaptive_thresholds = memory_data.get("adaptive_thresholds", self.adaptive_thresholds)
                self.query_patterns = defaultdict(int, memory_data.get("query_patterns", {}))
                
                # Rebuild experience index
                self.experience_index.clear()
                for i, exp in enumerate(self.experiences):
                    self.experience_index[exp.query_pattern].append(i)
                
                logger.info(f"Loaded agent memory with {len(self.experiences)} experiences")
            except Exception as e:
                logger.warning(f"Could not load agent memory: {e}")


class GoalManagementSystem:
    """Manages multi-step goals and task decomposition"""
    
    def __init__(self, storage_path: str = "./data/agent_goals"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.active_goals: Dict[str, AgentGoal] = {}
        self.completed_goals: List[AgentGoal] = []
        
        self.load_goals()
    
    def create_goal(self, description: str, criteria: List[str] = None) -> AgentGoal:
        """Create a new goal"""
        goal = AgentGoal(
            id=str(uuid.uuid4()),
            description=description,
            created_at=datetime.now().isoformat(),
            state=GoalState.ACTIVE,
            completion_criteria=criteria or []
        )
        
        self.active_goals[goal.id] = goal
        self.save_goals()
        return goal
    
    def decompose_goal(self, goal_id: str, subtasks: List[str]):
        """Break down a goal into subtasks"""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            for task in subtasks:
                goal.subtasks.append({
                    "description": task,
                    "completed": False,
                    "timestamp": None
                })
            self.save_goals()
    
    def update_progress(self, goal_id: str, subtask_idx: int = None):
        """Update goal progress"""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            
            if subtask_idx is not None and subtask_idx < len(goal.subtasks):
                goal.subtasks[subtask_idx]["completed"] = True
                goal.subtasks[subtask_idx]["timestamp"] = datetime.now().isoformat()
            
            # Calculate progress
            if goal.subtasks:
                completed = sum(1 for task in goal.subtasks if task["completed"])
                goal.progress = completed / len(goal.subtasks)
            
            # Check if goal is completed
            if goal.progress >= 1.0:
                goal.state = GoalState.COMPLETED
                self.completed_goals.append(goal)
                del self.active_goals[goal_id]
            
            self.save_goals()
    
    def get_relevant_goals(self, query: str) -> List[AgentGoal]:
        """Find goals relevant to current query"""
        relevant = []
        query_lower = query.lower()
        
        for goal in self.active_goals.values():
            # Simple relevance check - can be enhanced
            if any(word in query_lower for word in goal.description.lower().split()):
                relevant.append(goal)
        
        return relevant
    
    def save_goals(self):
        """Persist goals to disk"""
        goals_data = {
            "active": {gid: asdict(goal) for gid, goal in self.active_goals.items()},
            "completed": [asdict(goal) for goal in self.completed_goals[-100:]]  # Keep last 100
        }
        
        goals_file = os.path.join(self.storage_path, "goals.json")
        try:
            with open(goals_file, 'w') as f:
                json.dump(goals_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving goals: {e}")
    
    def load_goals(self):
        """Load goals from disk"""
        goals_file = os.path.join(self.storage_path, "goals.json")
        if os.path.exists(goals_file):
            try:
                with open(goals_file, 'r') as f:
                    goals_data = json.load(f)
                
                for gid, goal_dict in goals_data.get("active", {}).items():
                    goal = AgentGoal(**goal_dict)
                    goal.state = GoalState(goal.state)
                    self.active_goals[gid] = goal
                
                for goal_dict in goals_data.get("completed", []):
                    goal = AgentGoal(**goal_dict)
                    goal.state = GoalState(goal.state)
                    self.completed_goals.append(goal)
                
                logger.info(f"Loaded {len(self.active_goals)} active goals")
            except Exception as e:
                logger.warning(f"Could not load goals: {e}")


class ProactiveAssistant:
    """Provides proactive suggestions and anticipatory behavior"""
    
    def __init__(self):
        self.conversation_context: deque = deque(maxlen=10)
        self.topic_tracker: Dict[str, int] = defaultdict(int)
        self.suggestion_history: List[Dict[str, Any]] = []
    
    def analyze_context(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Analyze conversation context to generate proactive suggestions"""
        suggestions = []
        
        if not messages:
            return suggestions
        
        # Track topics
        for msg in messages[-5:]:
            if "user" in msg:
                self._track_topics(msg["user"])
        
        # Identify patterns
        last_message = messages[-1] if messages else {}
        
        # If discussing a topic repeatedly, suggest deeper exploration
        hot_topics = [topic for topic, count in self.topic_tracker.items() if count >= 2]
        if hot_topics:
            suggestions.append(f"Would you like me to provide more detailed information about {hot_topics[0]}?")
        
        # If user asked about implementation, suggest next steps
        if last_message.get("user", "").lower().startswith("how"):
            suggestions.append("I can help you create a step-by-step implementation plan if you'd like.")
        
        # If discussing problems, offer solutions
        problem_keywords = ["issue", "problem", "error", "doesn't work", "failed"]
        if any(kw in last_message.get("user", "").lower() for kw in problem_keywords):
            suggestions.append("Would you like me to help troubleshoot this issue systematically?")
        
        # Context-based suggestions
        if "sources" in last_message and last_message["sources"]:
            suggestions.append("I can search for additional sources or verify this information if needed.")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def anticipate_needs(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Anticipate user needs based on query pattern"""
        anticipated = {
            "pre_fetch": [],
            "prepare_tools": [],
            "related_queries": []
        }
        
        query_lower = query.lower()
        
        # Anticipate follow-up questions
        if "what is" in query_lower:
            anticipated["related_queries"].extend([
                "How does it work?",
                "What are the benefits?",
                "Can you provide examples?"
            ])
        
        if "how to" in query_lower:
            anticipated["related_queries"].extend([
                "What are the prerequisites?",
                "Are there alternatives?",
                "What are common pitfalls?"
            ])
        
        # Anticipate tool needs
        if any(word in query_lower for word in ["latest", "current", "recent"]):
            anticipated["prepare_tools"].append("web_search")
        
        if any(word in query_lower for word in ["document", "pdf", "file"]):
            anticipated["prepare_tools"].append("rag_search")
        
        return anticipated
    
    def _track_topics(self, text: str):
        """Track mentioned topics for pattern recognition"""
        # Simple topic extraction - can be enhanced with NLP
        words = text.lower().split()
        
        # Track nouns and important keywords
        important_words = [w for w in words if len(w) > 4 and w.isalpha()]
        for word in important_words:
            self.topic_tracker[word] += 1


class ConversationManager:
    """Enhanced conversation manager with goal tracking"""
    
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
            metadata={},
            active_goals=[],
            user_preferences={}
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
    """Enhanced RAG system with true agentic AI capabilities"""

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

        # Initialize AGENTIC components
        self.learning_system = AgentLearningSystem()
        self.goal_manager = GoalManagementSystem()
        self.proactive_assistant = ProactiveAssistant()
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(conversations_path)
        self.current_conversation: Optional[Conversation] = None

        # Initialize embedding model with optimizations
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            embed_batch_size=128,
            max_length=256,
            model_kwargs={
                "low_cpu_mem_usage": False,
                "device_map": None,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            },
        )

        # Configure LlamaIndex with optimized settings
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = None
        LlamaSettings.chunk_size = 1024
        LlamaSettings.chunk_overlap = 128

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

        # Adaptive routing knobs (now using learning system)
        self.always_enrich_modifications = True
        self.always_enrich_insufficient = True
        self.enrich_low_relevance_threshold = self.learning_system.adaptive_thresholds["relevance_threshold"]
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

        logger.info("AgenticRAG initialized with full agentic capabilities")

    def classify_query(self, query: str) -> QueryType:
        """Classify query type for adaptive routing"""
        query_lower = query.lower()
        
        # Check for specific patterns
        if any(term in query_lower for term in ["according to", "in the document", "pdf says"]):
            return QueryType.DOCUMENT_SPECIFIC
        
        if any(term in query_lower for term in ["latest", "current", "recent", "news", "today"]):
            return QueryType.WEB_CURRENT
        
        if self._is_modification_query(query):
            return QueryType.MODIFICATION
        
        if self._is_place_query(query):
            return QueryType.PLACE_INFO
        
        if self._is_open_ended(query):
            return QueryType.OPEN_ENDED
        
        # Check if needs both document and web
        doc_indicators = ["document", "file", "pdf", "uploaded"]
        web_indicators = ["search", "find", "online", "web"]
        
        has_doc = any(term in query_lower for term in doc_indicators)
        has_web = any(term in query_lower for term in web_indicators)
        
        if has_doc and has_web:
            return QueryType.HYBRID
        
        return QueryType.GENERAL_KNOWLEDGE

    def plan_approach(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Plan the approach for handling a query"""
        plan = {
            "steps": [],
            "tools_needed": [],
            "estimated_confidence": 0.7,
            "reasoning": []
        }
        
        # Check if we have learned strategy
        learned_strategy = self.learning_system.suggest_strategy(
            query, 
            {"query_type": query_type.value}
        )
        
        if learned_strategy and learned_strategy["confidence"] > 0.7:
            plan["steps"] = ["Use learned strategy"]
            plan["tools_needed"] = learned_strategy["tools"]
            plan["estimated_confidence"] = learned_strategy["confidence"]
            plan["reasoning"].append(learned_strategy["reasoning"])
            return plan
        
        # Otherwise, create new plan based on query type
        if query_type == QueryType.DOCUMENT_SPECIFIC:
            plan["steps"] = ["Search documents", "Extract relevant information", "Generate response"]
            plan["tools_needed"] = ["rag_search"]
            plan["reasoning"].append("Query specifically asks about document content")
            
        elif query_type == QueryType.WEB_CURRENT:
            plan["steps"] = ["Search web", "Analyze results", "Generate response"]
            plan["tools_needed"] = ["web_search"]
            plan["reasoning"].append("Query requires current information")
            
        elif query_type == QueryType.HYBRID:
            plan["steps"] = ["Search documents", "Search web", "Combine information", "Generate response"]
            plan["tools_needed"] = ["rag_search", "web_search"]
            plan["reasoning"].append("Query benefits from both document and web sources")
            
        elif query_type == QueryType.MODIFICATION:
            plan["steps"] = ["Understand context", "Search for variations", "Generate modifications"]
            plan["tools_needed"] = ["rag_search", "web_search"]
            plan["reasoning"].append("Query asks for modifications or alternatives")
            
        else:
            plan["steps"] = ["Analyze query", "Determine best source", "Generate response"]
            plan["tools_needed"] = ["general_knowledge"]
            plan["reasoning"].append("Query can be answered with general knowledge")
        
        return plan

    def self_evaluate_response(self, query: str, response: str, confidence: float) -> Dict[str, Any]:
        """Self-evaluate the quality of a response"""
        evaluation = {
            "quality_score": confidence,
            "completeness": True,
            "accuracy_confidence": confidence,
            "suggestions_for_improvement": []
        }
        
        # Check response length
        if len(response) < 100:
            evaluation["quality_score"] *= 0.8
            evaluation["suggestions_for_improvement"].append("Response might be too brief")
        
        # Check for uncertainty markers
        uncertainty_markers = ["might", "possibly", "perhaps", "unclear", "not sure"]
        if any(marker in response.lower() for marker in uncertainty_markers):
            evaluation["quality_score"] *= 0.9
            evaluation["accuracy_confidence"] *= 0.8
        
        # Check if response admits limitations
        if self.is_response_insufficient(response):
            evaluation["completeness"] = False
            evaluation["quality_score"] *= 0.7
            evaluation["suggestions_for_improvement"].append("Consider enriching with additional sources")
        
        return evaluation

    def process_with_goal_tracking(self, query: str) -> Optional[Dict[str, Any]]:
        """Process query in context of active goals"""
        relevant_goals = self.goal_manager.get_relevant_goals(query)
        
        if relevant_goals:
            goal = relevant_goals[0]  # Focus on most relevant goal
            
            # Check if this query contributes to a subtask
            for i, subtask in enumerate(goal.subtasks):
                if not subtask["completed"]:
                    # Simple matching - can be enhanced
                    if any(word in query.lower() for word in subtask["description"].lower().split()):
                        return {
                            "goal_id": goal.id,
                            "subtask_idx": i,
                            "goal_context": f"Working on goal: {goal.description}"
                        }
        
        return None

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
                    logger.info(f"Loaded metadata for {len(self.indexed_files)} indexed files")
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

                logger.info(f"Successfully loaded existing index with {len(self.indexed_files)} files")
                return True
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
        return False

    def _create_retriever(self):
        """Create or recreate the retriever"""
        if self.index is not None:
            # Adaptive similarity threshold from learning system
            similarity_cutoff = self.learning_system.adaptive_thresholds.get("relevance_threshold", 0.2)
            
            self.retriever = self.index.as_retriever(similarity_top_k=8)
            self.postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            logger.info(f"Retriever created with adaptive threshold: {similarity_cutoff}")

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

                if progress_callback:
                    progress_callback(30, "Creating vector index...")

                vector_store = ChromaVectorStore(chroma_collection=self.collection)
                self.storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

                self.index = VectorStoreIndex.from_documents(
                    all_documents,
                    storage_context=self.storage_context,
                    embed_model=self.embed_model,
                    show_progress=True,
                    insert_batch_size=32,
                )

                if progress_callback:
                    progress_callback(80, "Finalizing index...")

                for doc in all_documents:
                    self._update_document_metadata(doc)

            elif new_documents:
                logger.info("Adding new documents to existing index...")
                
                if progress_callback:
                    progress_callback(30, f"Adding {len(new_documents)} documents to index...")
                
                # Batch insert
                batch_size = 32
                for i in range(0, len(new_documents), batch_size):
                    batch = new_documents[i:i+batch_size]
                    for doc in batch:
                        self.index.insert(doc)
                    
                    if progress_callback:
                        progress = 30 + int((i / len(new_documents)) * 50)
                        progress_callback(progress, f"Indexed {min(i+batch_size, len(new_documents))}/{len(new_documents)} documents")

                for doc in new_documents:
                    self._update_document_metadata(doc)

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

    def _update_document_metadata(self, doc):
        """Update metadata for a document"""
        md = getattr(doc, "metadata", {}) or {}
        path = md.get("file_path") or md.get("filename") or md.get("file_name") or ""
        
        if path and not os.path.isabs(path):
            file_path = os.path.join(self.upload_path, os.path.basename(path))
        else:
            file_path = path
            
        file_name = os.path.basename(file_path) if file_path else getattr(doc, "doc_id", "unknown")
        file_hash = self._get_file_hash(file_path) if file_path else hashlib.md5(file_name.encode()).hexdigest()
        
        self.document_metadata[file_name] = {
            "loaded_at": datetime.now().isoformat(),
            "size": len(getattr(doc, "text", "") or ""),
            "hash": file_hash,
        }
        self.indexed_files.add(file_hash)

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for deduplication"""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()

    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform web search using Google CSE"""
        try:
            if not self.google_api_key or not self.google_cse_id:
                logger.warning("Google API key / CSE ID missing – web search disabled.")
                return []

            # Respect minimal delay between searches
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

            # Use adaptive threshold from learning system
            min_score = self.learning_system.adaptive_thresholds.get("relevance_threshold", 0.2)
            kept = [n for n in nodes if (n.score is None or n.score >= min_score)]
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
        """Check if Claude's response indicates insufficient information"""
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
        """Detect queries asking for modifications or variations"""
        if not text:
            return False
        t = text.lower()
        
        # Get conversation history
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
            ]
            
            for phrase in modification_phrases:
                if phrase in t:
                    return True
                    
        return False

    def _is_doc_only_intent(self, text: str) -> bool:
        """Detect when user wants answers ONLY from uploaded docs"""
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
        """Return True if the query looks like a place/attraction ask"""
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
        """Decision for enrichment using adaptive thresholds"""
        if self._is_doc_only_intent(q):
            return False
        
        # Use adaptive threshold from learning system
        relevance_threshold = self.learning_system.adaptive_thresholds.get("relevance_threshold", 0.35)
            
        if doc_relevance < relevance_threshold:
            logger.info(f"Enriching due to low relevance: {doc_relevance:.3f} < {relevance_threshold:.3f}")
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
        """Generate response using Claude API with enhanced prompting"""
        conversation_context = ""
        conversation_history = self.get_conversation_history()
        
        if conversation_history:
            recent_history = conversation_history[-3:]
            for h in recent_history:
                if 'user' in h and 'assistant' in h:
                    conversation_context += f"User: {h['user']}\nAssistant: {h['assistant']}\n\n"

        # Get user preferences from learning system
        user_prefs = self.learning_system.user_preferences
        style_instruction = ""
        if user_prefs.get("detail_level") == "brief":
            style_instruction = "Provide a concise response."
        elif user_prefs.get("detail_level") == "detailed":
            style_instruction = "Provide a comprehensive and detailed response."

        if tool_used == "rag_search":
            system_prompt = (
                f"You are an intelligent assistant that answers questions based on provided document context. "
                f"Use the context verbatim when possible. If the context doesn't fully answer, say what's missing. "
                f"Be honest about limitations - if information is incomplete, say so clearly. {style_instruction}"
            )
            user_prompt = (
                f"Based on the following context from the documents, answer this question: {query}\n\n"
                f"Document Context:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        elif tool_used == "web_search":
            system_prompt = (
                f"You are an intelligent assistant that answers questions using web search results.\n"
                f"You HAVE been provided search results; do not claim you cannot search the web.\n"
                f"If results are sparse, say what's missing, and still give best-effort guidance. {style_instruction}"
            )
            user_prompt = (
                f"Based on the following web search results, answer this question: {query}\n\n"
                f"Search Results:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        elif tool_used == "combined":
            system_prompt = (
                f"You are an intelligent assistant that combines information from documents and web search.\n"
                f"You HAVE been provided both; do not claim you cannot search the web.\n"
                f"Synthesize information from both sources to provide the most complete answer.\n"
                f"Clearly indicate which information comes from documents vs web when appropriate. {style_instruction}"
            )
            user_prompt = (
                f"Answer this question using both document context and web search results: {query}\n\n"
                f"Available Information:\n{context}\n\n"
                f"Previous conversation:\n{conversation_context}"
            )
        else:
            system_prompt = f"You are an intelligent, helpful assistant with broad knowledge. {style_instruction}"
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

        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            return (
                "🔐 **Authentication Error**\n\n"
                "Your API key appears to be invalid or has been revoked. "
                "Please check your API key and try again.\n\n"
                "To update your API key, click the 'Change API Key' button in the sidebar."
            )
        
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return f"I encountered an error generating a response: {str(e)}"

    def process_query(self, query: str) -> AgentResponse:
        """Enhanced agentic processing pipeline with learning and goal tracking"""
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        # Initialize reasoning chain
        reasoning_chain = []
        
        # Step 1: Classify query
        query_type = self.classify_query(query)
        reasoning_chain.append(f"Classified query as: {query_type.value}")
        
        # Step 2: Check for goal relevance
        goal_info = self.process_with_goal_tracking(query)
        if goal_info:
            reasoning_chain.append(goal_info["goal_context"])
        
        # Step 3: Plan approach
        plan = self.plan_approach(query, query_type)
        reasoning_chain.extend(plan["reasoning"])
        
        # Step 4: Get anticipatory insights
        anticipated = self.proactive_assistant.anticipate_needs(query, {"query_type": query_type})
        
        # Step 5: Execute plan
        all_contexts: List[str] = []
        all_sources: List[str] = []
        tools_used: List[str] = []
        
        # Get conversation history
        conversation_history = self.get_conversation_history()
        
        # Expand vague follow-ups
        effective_query = query
        if self._is_vague_followup(query) and conversation_history:
            prev_users = [h.get("user", "") for h in conversation_history if "user" in h]
            if prev_users:
                prev = prev_users[-1]
                if prev and prev.strip():
                    effective_query = f"{query} (context: {prev})"
        
        # Web-only flag
        web_only = self._is_web_only_intent(query)
        
        # Document search phase
        doc_context = None
        doc_relevance = 0.0
        doc_sources: List[str] = []
        force_web = False
        
        if not web_only and self.index is not None and self.retriever is not None and "rag_search" in plan["tools_needed"]:
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
                    # Check if sufficient
                    if doc_relevance > 0.5 and not self.is_response_insufficient(initial_response):
                        logger.info("Document search provided sufficient answer")
                        
                        # Self-evaluate
                        self_eval = self.self_evaluate_response(query, initial_response, doc_relevance)
                        
                        # Get proactive suggestions
                        suggestions = self.proactive_assistant.analyze_context(conversation_history + [{"user": query}])
                        
                        # Record experience
                        self.learning_system.record_experience(
                            query, "rag_only", ["rag_search"], 
                            doc_relevance, True,
                            {"query_type": query_type.value}
                        )
                        
                        # Update goal if relevant
                        if goal_info:
                            self.goal_manager.update_progress(goal_info["goal_id"], goal_info.get("subtask_idx"))
                        
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
                        
                        # Update performance metrics
                        response_time = time.time() - start_time
                        self.learning_system.performance.avg_response_time = (
                            (self.learning_system.performance.avg_response_time * 
                             self.learning_system.performance.total_queries + response_time) /
                            (self.learning_system.performance.total_queries + 1)
                        )
                        
                        return AgentResponse(
                            answer=initial_response,
                            sources=doc_sources,
                            confidence=max(doc_relevance, 0.7),
                            tool_used="rag_search",
                            metadata={
                                "timestamp": datetime.now().isoformat(),
                                "query_length": len(query),
                                "response_length": len(initial_response),
                                "response_time": response_time,
                            },
                            reasoning_chain=reasoning_chain,
                            suggestions=suggestions,
                            goal_progress={"goal_id": goal_info["goal_id"], "progress": self.goal_manager.active_goals[goal_info["goal_id"]].progress} if goal_info else None,
                            self_evaluation=self_eval
                        )
                    else:
                        force_web = True
        
        # Web search phase
        if "web_search" in plan["tools_needed"] or force_web:
            logger.info("Performing web search...")
            search_results = self.web_search(effective_query)
            
            if search_results:
                web_context = json.dumps(search_results, indent=2)
                all_contexts.append(f"Web Search Results:\n{web_context}")
                all_sources.extend([r["link"] for r in search_results[:3] if r.get("link")])
                tools_used.append("web_search")
        
        # Generate final response
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
            tools_used = ["general_knowledge"]
        
        # Adjust confidence if insufficient
        if self.is_response_insufficient(final_response):
            confidence = min(confidence, 0.55)
        
        # Self-evaluate
        self_eval = self.self_evaluate_response(query, final_response, confidence)
        
        # Get proactive suggestions
        suggestions = self.proactive_assistant.analyze_context(
            conversation_history + [{"user": query, "assistant": final_response}]
        )
        
        # Record experience
        success = confidence > 0.6 and not self.is_response_insufficient(final_response)
        self.learning_system.record_experience(
            query, tool_description, tools_used,
            confidence, success,
            {"query_type": query_type.value}
        )
        
        # Update goal if relevant
        if goal_info:
            self.goal_manager.update_progress(goal_info["goal_id"], goal_info.get("subtask_idx"))
        
        # Adapt thresholds periodically
        if self.learning_system.performance.total_queries % 20 == 0:
            self.learning_system.adapt_thresholds()
            self._create_retriever()  # Update retriever with new thresholds
        
        # Store in conversation
        if self.current_conversation:
            self.current_conversation.messages.append({
                "user": query,
                "assistant": final_response,
                "timestamp": datetime.now().isoformat(),
                "tool_used": " + ".join(tools_used),
                "sources": all_sources,
                "confidence": confidence
            })
            self.current_conversation.updated_at = datetime.now().isoformat()
            self.conversation_manager.save_conversation(self.current_conversation)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        return AgentResponse(
            answer=final_response,
            sources=all_sources,
            confidence=confidence,
            tool_used=" + ".join(tools_used),
            metadata={
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query),
                "response_length": len(final_response),
                "tools_used": tools_used,
                "doc_relevance": doc_relevance if doc_context else 0,
                "response_time": response_time,
            },
            reasoning_chain=reasoning_chain,
            suggestions=suggestions,
            goal_progress={"goal_id": goal_info["goal_id"], "progress": self.goal_manager.active_goals[goal_info["goal_id"]].progress} if goal_info else None,
            self_evaluation=self_eval
        )

    def provide_feedback(self, message_idx: int, feedback_type: str):
        """Process user feedback to improve future responses"""
        if self.current_conversation and message_idx < len(self.current_conversation.messages):
            message = self.current_conversation.messages[message_idx]
            
            context = {
                "response_length": "long" if len(message.get("assistant", "")) > 500 else "brief",
                "sources_used": message.get("sources", []),
                "tool_used": message.get("tool_used", "")
            }
            
            self.learning_system.learn_user_preference(feedback_type, context)
            
            # Update user satisfaction score
            if feedback_type == "positive":
                self.learning_system.performance.user_satisfaction_score = (
                    self.learning_system.performance.user_satisfaction_score * 0.95 + 1.0 * 0.05
                )
            else:
                self.learning_system.performance.user_satisfaction_score = (
                    self.learning_system.performance.user_satisfaction_score * 0.95 + 0.0 * 0.05
                )
            
            # Save learning after feedback
            self.learning_system.save_memory()
            
            logger.info(f"Received {feedback_type} feedback for message {message_idx}")

    def create_goal(self, description: str) -> str:
        """Create a new goal for the agent to pursue"""
        goal = self.goal_manager.create_goal(description)
        
        if self.current_conversation:
            self.current_conversation.active_goals.append(goal.id)
            self.conversation_manager.save_conversation(self.current_conversation)
        
        logger.info(f"Created new goal: {goal.id}")
        return goal.id

    def clear_conversation(self):
        """Clear current conversation"""
        if self.current_conversation:
            self.current_conversation.messages = []
            self.current_conversation.updated_at = datetime.now().isoformat()
            self.conversation_manager.save_conversation(self.current_conversation)
        logger.info("Conversation history cleared")

    def get_system_info(self) -> Dict[str, Any]:
        """Get enhanced system information with learning metrics"""
        conversation_count = len(self.get_conversation_history()) if self.current_conversation else 0
        
        # Get learning metrics
        learning_metrics = {
            "total_experiences": len(self.learning_system.experiences),
            "avg_confidence": self.learning_system.performance.avg_confidence,
            "success_rate": (self.learning_system.performance.successful_queries / 
                           max(1, self.learning_system.performance.total_queries)),
            "user_satisfaction": self.learning_system.performance.user_satisfaction_score,
            "adaptive_thresholds": self.learning_system.adaptive_thresholds
        }
        
        # Get active goals
        active_goals = [
            {
                "id": goal.id,
                "description": goal.description,
                "progress": goal.progress
            }
            for goal in self.goal_manager.active_goals.values()
        ]
        
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
            "learning_metrics": learning_metrics,
            "active_goals": active_goals,
            "total_queries_processed": self.learning_system.performance.total_queries,
        }