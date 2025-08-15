import streamlit as st
import os
from dotenv import load_dotenv
import sys
import time
from pathlib import Path
import hashlib
from datetime import datetime

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

from app.agents.rag_agent import AgenticRAG, AgentResponse
import json

# Load environment variables
load_dotenv()

# PRAGYA Branding with India Flag Colors (Sharp & Clear)
PROJECT_NAME = "PRAGYA"
PROJECT_TAGLINE = "Persistent Retrieval Augmented Generation Your Assistant"
PROJECT_HINDI = "प्रज्ञा - Your Intelligent Knowledge Companion"
PROJECT_VERSION = "v2.2"
DEVELOPER = "Udish Kumar"

# India Flag Colors (Official)
SAFFRON = "#FF9933"  # Indian Flag Saffron
WHITE = "#FFFFFF"    # White
GREEN = "#138808"    # Indian Flag Green
NAVY_BLUE = "#000080" # Ashoka Chakra Blue

# Page configuration
st.set_page_config(
    page_title=f"{PROJECT_NAME} - Intelligent RAG Assistant",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Modern Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* PRAGYA Header with Crisp Tricolor Gradient */
    .pragya-header {
        font-weight: 800;
        font-size: 2.5em;
        text-align: center;
        padding: 20px 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        position: relative;
        display: inline-block;
        width: 100%;
    }
    
    /* Additional Flag Elements for clear visibility */
    .tricolor-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin: 0 4px;
    }
    
    .badge-saffron {
        background: #FF9933;
        color: white;
    }
    
    .badge-white {
        background: #FFFFFF;
        color: #333;
        border: 1px solid #ddd;
    }
    
    .badge-green {
        background: #138808;
        color: white;
    }
    
    /* Clear letter styling without gradient issues */
    .letter-saffron {
        color: #FF9933;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .letter-white {
        color: #666;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .letter-green {
        color: #138808;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Indian Flag Visual Bar */
    .flag-bar {
        display: flex;
        height: 6px;
        width: 80%;
        margin: 10px auto;
        border-radius: 3px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .flag-bar .saffron {
        flex: 1;
        background: #FF9933;
    }
    
    .flag-bar .white {
        flex: 1;
        background: #FFFFFF;
        border-top: 1px solid #eee;
        border-bottom: 1px solid #eee;
    }
    
    .flag-bar .green {
        flex: 1;
        background: #138808;
    }
    
    /* Enhanced subtitle with better contrast */
    .pragya-subtitle {
        font-family: 'Noto Sans Devanagari', 'Inter', sans-serif;
        color: #555;
        text-align: center;
        font-size: 1.15em;
        margin-top: 5px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced Card Design */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #FF9933;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Status Badge Styles */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Enhanced Conversation List */
    .conversation-item {
        background: white;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e5e7eb;
        position: relative;
        overflow: hidden;
    }
    
    .conversation-item::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: linear-gradient(180deg, #FF9933 0%, #138808 100%);
        opacity: 0;
        transition: opacity 0.2s ease;
    }
    
    .conversation-item:hover {
        background: #f9fafb;
        transform: translateX(3px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .conversation-item:hover::before {
        opacity: 1;
    }
    
    .conversation-active {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border-color: #FF9933;
    }
    
    /* Enhanced Button Styles */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Primary Action Buttons */
    div[data-testid="column"]:has(button:contains("New Chat")) button,
    div[data-testid="column"]:has(button:contains("History")) button {
        background: linear-gradient(135deg, #FF9933 0%, #ff8000 100%);
        color: white;
        border: none;
    }
    
    /* Tool Status Grid */
    .tool-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    
    .tool-item {
        background: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .tool-item:hover {
        background: #f9fafb;
        border-color: #FF9933;
    }
    
    /* Enhanced Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        margin: 12px 0;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Metadata Pills */
    .metadata-pill {
        display: inline-block;
        padding: 4px 10px;
        background: #f3f4f6;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 8px;
        color: #4b5563;
    }
    
    /* Enhanced File Upload Area */
    .uploadedFile {
        border-radius: 8px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
    }
    
    /* Progress Bar Enhancement */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FF9933 0%, #138808 100%);
    }
    
    /* Sidebar Enhancements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f5 100%);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f9fafb;
        border-color: #FF9933;
    }
    
    /* Input Field Enhancement */
    .stTextInput > div > div > input,
    .stChatInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInput > div > div > input:focus {
        border-color: #FF9933;
        box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.1);
    }
    
    /* Tooltip Enhancement */
    [data-baseweb="tooltip"] {
        border-radius: 8px;
        background: #1f2937;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .pragya-header {
            font-size: 1.8em;
        }
        
        .tool-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Smooth Scrolling */
    .main {
        scroll-behavior: smooth;
    }
    
    /* Enhanced Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your_actual_claude_api_key_here':
        st.error("⚠️ Please add your ANTHROPIC_API_KEY to the .env file!")
        st.info("Get your API key from: https://console.anthropic.com")
        st.stop()

    with st.spinner(f"🕉️ Initializing {PROJECT_NAME}..."):
        try:
            st.session_state.agent = AgenticRAG(
                anthropic_api_key=api_key,
                vector_store_path=os.getenv('VECTOR_STORE_PATH', './data/vector_store'),
                upload_path=os.getenv('UPLOAD_PATH', './data/uploads'),
                conversations_path=os.getenv('CONVERSATIONS_PATH', './data/conversations')
            )
            
            # Check if there are existing documents to load
            upload_path = os.getenv('UPLOAD_PATH', './data/uploads')
            if os.path.exists(upload_path):
                pdf_files = [f for f in os.listdir(upload_path) if f.endswith('.pdf')]
                if pdf_files:
                    with st.spinner(f"📚 Loading {len(pdf_files)} existing documents..."):
                        success = st.session_state.agent.load_documents()
                        if success:
                            st.success(f"✅ Loaded {len(pdf_files)} documents from previous session")
            
            # Initialize or load conversation
            conversations = st.session_state.agent.conversation_manager.list_conversations()
            if conversations:
                # Load the most recent conversation
                latest_conv = st.session_state.agent.conversation_manager.load_conversation(conversations[0]['id'])
                st.session_state.agent.set_current_conversation(latest_conv)
                st.session_state.current_conversation_id = latest_conv.id
            else:
                # Create a new conversation
                new_conv = st.session_state.agent.conversation_manager.create_conversation()
                st.session_state.agent.set_current_conversation(new_conv)
                st.session_state.current_conversation_id = new_conv.id
                
        except Exception as e:
            st.error(f"Failed to initialize PRAGYA: {str(e)}")
            st.info("Try clearing the cache: rm -rf ~/.cache/huggingface/hub/")
            st.stop()

# Initialize other session states
if 'messages' not in st.session_state:
    # Load messages from current conversation if exists
    if st.session_state.agent.current_conversation:
        st.session_state.messages = st.session_state.agent.current_conversation.messages
    else:
        st.session_state.messages = []

if 'current_conversation_id' not in st.session_state:
    if st.session_state.agent.current_conversation:
        st.session_state.current_conversation_id = st.session_state.agent.current_conversation.id
    else:
        st.session_state.current_conversation_id = None

if 'uploaded_files_hash' not in st.session_state:
    st.session_state.uploaded_files_hash = None

if 'processing_documents' not in st.session_state:
    st.session_state.processing_documents = False

if 'last_upload_count' not in st.session_state:
    st.session_state.last_upload_count = 0

if 'show_conversation_list' not in st.session_state:
    st.session_state.show_conversation_list = False

# Helper function to get files hash
def get_files_hash(files):
    """Create a hash of uploaded files to detect changes"""
    if not files:
        return None
    hash_obj = hashlib.md5()
    for file in files:
        hash_obj.update(file.name.encode())
        hash_obj.update(str(file.size).encode())
    return hash_obj.hexdigest()

# Helper function to switch conversation
def switch_conversation(conv_id):
    """Switch to a different conversation"""
    conversation = st.session_state.agent.conversation_manager.load_conversation(conv_id)
    if conversation:
        st.session_state.agent.set_current_conversation(conversation)
        st.session_state.current_conversation_id = conv_id
        st.session_state.messages = conversation.messages
        st.rerun()

# Helper function to create new conversation
def create_new_conversation():
    """Create a new conversation"""
    new_conv = st.session_state.agent.conversation_manager.create_conversation()
    st.session_state.agent.set_current_conversation(new_conv)
    st.session_state.current_conversation_id = new_conv.id
    st.session_state.messages = []
    st.rerun()

# Helper function to delete conversation
def delete_conversation(conv_id):
    """Delete a conversation"""
    if conv_id == st.session_state.current_conversation_id:
        # If deleting current conversation, create a new one
        create_new_conversation()
    else:
        st.session_state.agent.conversation_manager.delete_conversation(conv_id)
        st.rerun()

# Sidebar with Enhanced Design
with st.sidebar:
    # PRAGYA Branding with crisp tricolor theme
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 2.5em; font-weight: 800; letter-spacing: 3px; padding: 15px 10px;">
            <span style="color: #FF9933;">P</span>
            <span style="color: #FF9933;">R</span>
            <span style="color: #FF9933;">A</span>
            <span style="color: #666;">G</span>
            <span style="color: #138808;">Y</span>
            <span style="color: #138808;">A</span>
            <span style="margin-left: 10px;">🧘</span>
        </div>
        <div class="flag-bar">
            <div class="saffron"></div>
            <div class="white"></div>
            <div class="green"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<p class="pragya-subtitle">{PROJECT_HINDI}</p>', unsafe_allow_html=True)
    
    # Elegant divider
    st.markdown("---")

    # Conversation Management Section with Better Design
    st.markdown("### 💬 Conversations")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_conversation()
    
    with col2:
        history_label = "📋 Hide" if st.session_state.show_conversation_list else "📋 History"
        if st.button(history_label, use_container_width=True):
            st.session_state.show_conversation_list = not st.session_state.show_conversation_list
    
    # Enhanced Conversation List
    if st.session_state.show_conversation_list:
        conversations = st.session_state.agent.conversation_manager.list_conversations()
        
        if conversations:
            st.caption(f"📂 {len(conversations)} saved conversations")
            
            # Create a scrollable area with enhanced styling
            conversation_container = st.container()
            with conversation_container:
                for conv in conversations[:10]:  # Show latest 10
                    conv_date = datetime.fromisoformat(conv['updated_at']).strftime('%b %d, %H:%M')
                    is_current = conv['id'] == st.session_state.current_conversation_id
                    
                    # Enhanced conversation item display
                    if is_current:
                        st.markdown(f"""
                        <div class="conversation-item conversation-active">
                            <strong>✓ {conv['title'][:25]}...</strong><br>
                            <small style="color: #666;">{conv_date}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            if st.button(f"💭 {conv['title'][:25]}...\n{conv_date}", 
                                       key=f"conv_{conv['id']}", 
                                       use_container_width=True):
                                switch_conversation(conv['id'])
                        with col2:
                            if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                                delete_conversation(conv['id'])
        else:
            st.info("💭 No saved conversations yet")
    
    # Current Conversation Info with Enhanced Card Design
    if st.session_state.agent.current_conversation:
        current_conv = st.session_state.agent.current_conversation
        with st.expander("📍 Current Session", expanded=False):
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.text(f"📝 {current_conv.title[:30]}")
            st.text(f"💬 {len(current_conv.messages)} messages")
            created = datetime.fromisoformat(current_conv.created_at)
            st.text(f"📅 {created.strftime('%b %d, %Y %H:%M')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Rename conversation with better styling
            new_title = st.text_input("✏️ Rename:", 
                                     value=current_conv.title, 
                                     key="rename_conv",
                                     placeholder="Enter new title...")
            if new_title != current_conv.title:
                st.session_state.agent.conversation_manager.update_conversation_title(current_conv.id, new_title)
                st.success("✅ Renamed successfully!")

    st.markdown("---")

    # Enhanced Document Management Section
    st.markdown("### 📄 Document Library")

    # Get current system info for display
    system_info = st.session_state.agent.get_system_info()

    # Document Status Card
    if system_info['documents_loaded'] > 0:
        st.markdown(f'<span class="status-badge status-ready">✅ {system_info["documents_loaded"]} documents ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">⚠️ No documents loaded</span>', unsafe_allow_html=True)

    # Enhanced File Upload
    with st.container():
        uploaded_files = st.file_uploader(
            "📤 Upload PDFs",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload PDF documents for intelligent Q&A",
            key="pdf_uploader"
        )

    # Process uploaded files with better feedback
    current_hash = get_files_hash(uploaded_files)

    if uploaded_files and current_hash != st.session_state.uploaded_files_hash and not st.session_state.processing_documents:
        st.session_state.processing_documents = True
        st.session_state.uploaded_files_hash = current_hash

        upload_path = os.getenv('UPLOAD_PATH', './data/uploads')
        os.makedirs(upload_path, exist_ok=True)

        # Save files with enhanced progress
        saved_files = []
        new_files = []

        progress_container = st.container()
        with progress_container:
            with st.spinner("📥 Processing documents..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(upload_path, uploaded_file.name)

                    file_exists = os.path.exists(file_path)
                    if file_exists:
                        with open(file_path, 'rb') as f:
                            existing_content = f.read()
                        new_content = bytes(uploaded_file.getbuffer())
                        if existing_content == new_content:
                            st.info(f"📄 {uploaded_file.name[:20]}... already exists")
                            continue

                    with open(file_path, "wb") as f:
                        f.write(bytes(uploaded_file.getbuffer()))
                    saved_files.append(uploaded_file.name)
                    new_files.append(uploaded_file.name)
                    st.success(f"✅ Saved {uploaded_file.name[:20]}...")

        # Index documents with enhanced progress
        if new_files:
            progress_bar = st.progress(0, text="🔄 Starting indexing...")
            
            def progress_callback(progress, message):
                progress_bar.progress(progress / 100, text=f"⚡ {message}")
            
            success = st.session_state.agent.load_documents(
                force_reload=False,
                progress_callback=progress_callback
            )
            
            if success:
                st.balloons()
                st.success(f"🎉 Successfully indexed {len(new_files)} documents!")
            else:
                st.error("❌ Indexing failed. Please try again.")

        st.session_state.processing_documents = False
        time.sleep(1)
        st.rerun()

    # Enhanced System Status Section
    st.markdown("---")
    st.markdown("### 📊 System Status")

    # Status Metrics with Cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Documents", system_info['documents_loaded'], 
                 delta=None if system_info['documents_loaded'] == 0 else "Active")
    with col2:
        st.metric("💬 Messages", system_info['conversation_length'])

    # Index Status with Visual Indicator
    if system_info['index_ready']:
        st.markdown('<span class="status-badge status-ready">✅ System Ready</span>', unsafe_allow_html=True)
        with st.expander("📊 Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📁 Files", system_info.get('indexed_files_count', 0))
            with col2:
                st.metric("🗄️ Vectors", system_info.get('collection_count', 0))
    else:
        st.markdown('<span class="status-badge status-warning">⚠️ No index</span>', unsafe_allow_html=True)
        st.caption("Upload PDFs to enable document search")

    # Document List with Enhanced Display
    if system_info['document_list']:
        with st.expander(f"📁 Document Library ({len(system_info['document_list'])})", expanded=False):
            for doc in system_info['document_list']:
                doc_name = doc if len(doc) <= 25 else doc[:22] + "..."
                st.markdown(f"📄 **{doc_name}**")

    # Enhanced Tool Status Grid
    st.markdown("---")
    with st.expander("🛠️ Available Tools", expanded=False):
        st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
        
        tools_status = [
            ("📚 RAG Search", "Active" if system_info['index_ready'] else "Inactive"),
            ("🌐 Web Search", "Active"),
            ("🤖 Claude AI", "Active"),
            ("🗄️ Vector DB", "Active")
        ]
        
        col1, col2 = st.columns(2)
        for i, (tool, status) in enumerate(tools_status):
            with col1 if i % 2 == 0 else col2:
                if status == "Active":
                    st.success(f"{tool}\n✅ {status}")
                else:
                    st.warning(f"{tool}\n⚠️ {status}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Action Buttons with Enhanced Styling
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear current conversation"):
            st.session_state.messages = []
            st.session_state.agent.clear_conversation()
            st.rerun()

    with col2:
        if st.button("🔄 Re-index", use_container_width=True, help="Rebuild document index"):
            progress_bar = st.progress(0, text="🔄 Re-indexing documents...")
            
            def progress_callback(progress, message):
                progress_bar.progress(progress / 100, text=f"⚡ {message}")
            
            success = st.session_state.agent.load_documents(
                force_reload=True,
                progress_callback=progress_callback
            )
            
            if success:
                st.success("✅ Re-indexing complete!")
            else:
                st.error("❌ Re-indexing failed")
            time.sleep(1)
            st.rerun()

    # Enhanced About Section with Tricolor Theme
    st.markdown("---")
    with st.expander("ℹ️ About PRAGYA", expanded=False):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fff9f4 0%, #ffffff 50%, #f4fff4 100%); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid #FF9933; border-right: 4px solid #138808;">
        
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 1.5em; font-weight: 800;">
                <span style="color: {SAFFRON};">P</span>
                <span style="color: {SAFFRON};">R</span>
                <span style="color: {SAFFRON};">A</span>
                <span style="color: #666;">G</span>
                <span style="color: {GREEN};">Y</span>
                <span style="color: {GREEN};">A</span>
            </span>
            <span style="margin-left: 10px;">🧘</span>
        </div>
        
        **Sanskrit:** प्रज्ञा (Prajñā) - Wisdom, Intelligence, Understanding  
        
        **✨ Core Features:**  
        📚 **Document RAG** - Intelligent search across your PDFs  
        🌐 **Web Search** - Real-time information retrieval  
        🤖 **Claude AI** - Advanced language understanding  
        🔄 **Smart Routing** - Automatic source selection  
        💾 **Persistent Memory** - Knowledge preservation  
        💬 **Multi-conversation** - Manage multiple chat sessions  
        
        **🎯 Use Cases:**
        • Research & Analysis
        • Document Q&A
        • Knowledge Management
        • Information Synthesis
        
        <div class="flag-bar" style="margin: 15px 0;">
            <div class="saffron"></div>
            <div class="white"></div>
            <div class="green"></div>
        </div>
        
        **Developer:** {DEVELOPER}  
        **Version:** {PROJECT_VERSION}  
        **Built with:** Streamlit, Claude AI, ChromaDB
        </div>
        """, unsafe_allow_html=True)

# Main Chat Interface with Enhanced Design
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <div style="font-size: 3em; font-weight: 800; letter-spacing: 4px; padding: 20px 10px;">
        <span style="color: #FF9933; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">🧘 P</span>
        <span style="color: #FF9933; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">R</span>
        <span style="color: #FF9933; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">A</span>
        <span style="color: #666; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">G</span>
        <span style="color: #138808; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">Y</span>
        <span style="color: #138808; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">A</span>
    </div>
    <div class="flag-bar">
        <div class="saffron"></div>
        <div class="white"></div>
        <div class="green"></div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown(f'<p class="pragya-subtitle">{PROJECT_TAGLINE}</p>', unsafe_allow_html=True)

# Enhanced Capability Display with Tricolor Theme
system_info = st.session_state.agent.get_system_info()

# Tricolor accent bar
st.markdown("""
<div class="flag-bar" style="margin-bottom: 15px;">
    <div class="saffron"></div>
    <div class="white"></div>
    <div class="green"></div>
</div>
""", unsafe_allow_html=True)

capability_cols = st.columns(4)

with capability_cols[0]:
    if system_info['index_ready']:
        st.markdown(f'<span class="status-badge status-ready">📚 {system_info["documents_loaded"]} docs</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">📄 No docs</span>', unsafe_allow_html=True)

with capability_cols[1]:
    st.markdown('<span class="status-badge status-ready">🌐 Web</span>', unsafe_allow_html=True)

with capability_cols[2]:
    st.markdown('<span class="status-badge status-ready">🤖 AI</span>', unsafe_allow_html=True)

with capability_cols[3]:
    conv_title = st.session_state.agent.current_conversation.title[:15] if st.session_state.agent.current_conversation else 'New'
    st.markdown(f'<span class="status-badge status-ready">💬 {conv_title}</span>', unsafe_allow_html=True)

# Display chat messages with enhanced styling
for message in st.session_state.messages:
    role = message.get("role")
    
    # Handle both old format (user/assistant keys) and new format (role key)
    if role:
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.get("content", message.get("user", "")))
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🧘"):
                st.markdown(message.get("content", message.get("assistant", "")))
                
                # Enhanced metadata display
                metadata_container = st.container()
                with metadata_container:
                    metadata_items = []
                    
                    if "tool_used" in message:
                        tool_emoji = {
                            "rag_search": "📚",
                            "web_search": "🌐",
                            "general_knowledge": "🧠",
                            "rag_search + web_search": "📚🌐",
                            "combined": "📚🌐"
                        }.get(message['tool_used'], "🛠️")
                        metadata_items.append(f"{tool_emoji} {message['tool_used']}")
                    
                    if "confidence" in message:
                        confidence = message.get('confidence', 0)
                        if confidence >= 0.8:
                            metadata_items.append(f"✅ {confidence:.0%}")
                        elif confidence >= 0.6:
                            metadata_items.append(f"📊 {confidence:.0%}")
                        else:
                            metadata_items.append(f"⚠️ {confidence:.0%}")
                    
                    if "timestamp" in message:
                        try:
                            dt = datetime.fromisoformat(message.get('timestamp', ''))
                            metadata_items.append(f"⏱️ {dt.strftime('%H:%M')}")
                        except:
                            pass
                    
                    if "sources" in message and message["sources"]:
                        metadata_items.append(f"📚 {len(message['sources'])} sources")
                    
                    # Display metadata pills
                    if metadata_items:
                        st.caption(" • ".join(metadata_items))
                    
                    # Sources popover
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 View Sources", expanded=False):
                            for i, source in enumerate(message["sources"], 1):
                                if isinstance(source, str) and source.startswith("http"):
                                    domain = source.split('/')[2] if len(source.split('/')) > 2 else source[:30]
                                    st.markdown(f"{i}. 🔗 [{domain}]({source})")
                                else:
                                    source_name = source[:50] + "..." if len(source) > 50 else source
                                    st.markdown(f"{i}. 📄 {source_name}")

# Enhanced Chat Input
if prompt := st.chat_input(f"Ask {PROJECT_NAME} anything... (प्रज्ञा से कुछ भी पूछें)", key="chat_input"):
    if st.session_state.processing_documents:
        st.warning("⏳ Please wait, documents are being indexed...")
    else:
        # Add user message with animation
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧘"):
            message_placeholder = st.empty()
            metadata_placeholder = st.empty()

            # Enhanced thinking status
            with st.status(f"🧘 {PROJECT_NAME} is contemplating...", expanded=True) as status:
                status.update(label="🔍 Analyzing query...", state="running")
                response: AgentResponse = st.session_state.agent.process_query(prompt)
                status.update(label="✨ Wisdom delivered!", state="complete", expanded=False)

            # Display response with animation
            message_placeholder.markdown(response.answer)

            # Enhanced metadata display
            with metadata_placeholder.container():
                metadata_items = []
                
                tool_emoji = {
                    "rag_search": "📚",
                    "web_search": "🌐",
                    "general_knowledge": "🧠",
                    "rag_search + web_search": "📚🌐",
                    "combined": "📚🌐"
                }.get(response.tool_used, "🛠️")
                metadata_items.append(f"{tool_emoji} {response.tool_used}")
                
                if response.confidence >= 0.8:
                    metadata_items.append(f"✅ {response.confidence:.0%}")
                elif response.confidence >= 0.6:
                    metadata_items.append(f"📊 {response.confidence:.0%}")
                else:
                    metadata_items.append(f"⚠️ {response.confidence:.0%}")
                
                metadata_items.append(f"⏱️ {datetime.now().strftime('%H:%M')}")
                
                if response.sources:
                    metadata_items.append(f"📚 {len(response.sources)} sources")
                
                st.caption(" • ".join(metadata_items))
                
                # Sources display
                if response.sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(response.sources, 1):
                            if isinstance(source, str) and source.startswith("http"):
                                domain = source.split('/')[2] if len(source.split('/')) > 2 else source[:30]
                                st.markdown(f"{i}. 🔗 [{domain}]({source})")
                            else:
                                source_name = source[:50] + "..." if len(source) > 50 else source
                                st.markdown(f"{i}. 📄 {source_name}")

            # Save to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "tool_used": response.tool_used,
                "confidence": response.confidence,
                "timestamp": datetime.now().isoformat()
            })

# Enhanced Welcome Screen
if not st.session_state.messages:
    st.markdown("---")
    
    # Welcome message with gradient and clear tricolor
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #333; font-weight: 700;">
            <span style="color: #FF9933;">🙏</span> Namaste! Welcome to 
            <span class="letter-saffron">P</span><span class="letter-saffron">R</span><span class="letter-saffron">A</span><span class="letter-white">G</span><span class="letter-green">Y</span><span class="letter-green">A</span>
        </h2>
        <p style="color: #666; font-size: 1.1em;">Your intelligent companion for knowledge and wisdom</p>
        <div style="margin: 15px auto; font-size: 1.5em;">
            <span style="color: #FF9933;">■</span>
            <span style="color: #FFFFFF; text-shadow: 0 0 2px #ccc;">■</span>
            <span style="color: #138808;">■</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive prompt cards
    st.markdown("### 💡 Try asking me about:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📚 Documents</h4>
            <ul style="list-style: none; padding: 0;">
                <li>• Summarize key points</li>
                <li>• Find specific information</li>
                <li>• Compare sections</li>
                <li>• Extract insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🌐 Current Info</h4>
            <ul style="list-style: none; padding: 0;">
                <li>• Latest AI news</li>
                <li>• Tech trends</li>
                <li>• Recent research</li>
                <li>• Market updates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 Knowledge</h4>
            <ul style="list-style: none; padding: 0;">
                <li>• Complex concepts</li>
                <li>• Technical topics</li>
                <li>• Best practices</li>
                <li>• How-to guides</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Feature highlight with tricolor accent
    st.markdown("""
    <div style="padding: 10px; margin-top: 20px;">
        <div class="flag-bar" style="width: 100%; margin-bottom: 15px;">
            <div class="saffron"></div>
            <div class="white"></div>
            <div class="green"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    💡 **Pro Tip:** PRAGYA (प्रज्ञा) intelligently selects the best source for your answer - 
    combining document search, web intelligence, and AI knowledge to deliver comprehensive insights!
    """)

# Enhanced Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; 
            background: linear-gradient(135deg, #fff9f4 0%, #ffffff 50%, #f4fff4 100%); 
            padding: 20px; 
            border-radius: 10px; 
            margin-top: 20px;
            border: 2px solid transparent;
            border-image: linear-gradient(90deg, #FF9933, #FFFFFF, #138808) 1;'>
    <div style="font-size: 1.4em; font-weight: 700; letter-spacing: 2px; margin-bottom: 10px;">
        <span style="color: #FF9933;">P</span>
        <span style="color: #FF9933;">R</span>
        <span style="color: #FF9933;">A</span>
        <span style="color: #666;">G</span>
        <span style="color: #138808;">Y</span>
        <span style="color: #138808;">A</span>
    </div>
    <p style='color: #555; font-weight: 600; margin: 5px;'>
        {PROJECT_HINDI}
    </p>
    <div class="flag-bar" style="width: 40%; margin: 15px auto;">
        <div class="saffron"></div>
        <div class="white"></div>
        <div class="green"></div>
    </div>
    <p style='color: #666; margin: 10px 0 5px 0;'>
        <strong>Powered by:</strong> Claude AI 🤖 | ChromaDB 🗄️ | Google Search 🔎
    </p>
    <p style='color: #888; font-size: 0.9em; margin-top: 10px;'>
        Crafted with ❤️ by {DEVELOPER} | {PROJECT_VERSION}
    </p>
</div>
""", unsafe_allow_html=True)