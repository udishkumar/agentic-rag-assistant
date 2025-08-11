import streamlit as st
import os
from dotenv import load_dotenv
import sys
import time
from pathlib import Path
import hashlib

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

from app.agents.rag_agent import AgenticRAG, AgentResponse
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .chat-message { padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; }
    .user-message { background-color: #e3f2fd; margin-left: 20%; }
    .assistant-message { background-color: #ffffff; margin-right: 20%; border: 1px solid #e0e0e0; }
    .source-card { background-color: #f5f5f5; padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.25rem; font-size: 0.9rem; }
    .metric-card { background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    /* Hide the status widget that appears during indexing */
    div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your_actual_claude_api_key_here':
        st.error("⚠️ Please add your ANTHROPIC_API_KEY to the .env file!")
        st.info("Get your API key from: https://console.anthropic.com")
        st.stop()

    with st.spinner("🚀 Initializing RAG system..."):
        st.session_state.agent = AgenticRAG(
            anthropic_api_key=api_key,
            vector_store_path=os.getenv('VECTOR_STORE_PATH', './data/vector_store'),
            upload_path=os.getenv('UPLOAD_PATH', './data/uploads')
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

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'uploaded_files_hash' not in st.session_state:
    st.session_state.uploaded_files_hash = None

if 'processing_documents' not in st.session_state:
    st.session_state.processing_documents = False

if 'last_upload_count' not in st.session_state:
    st.session_state.last_upload_count = 0

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

# Sidebar
with st.sidebar:
    st.title("🤖 Agentic RAG Assistant")
    st.markdown("---")

    # File upload section
    st.header("📄 Document Upload")

    # Get current system info for display
    system_info = st.session_state.agent.get_system_info()

    # Show current document status
    if system_info['documents_loaded'] > 0:
        st.success(f"✅ {system_info['documents_loaded']} documents indexed")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        accept_multiple_files=True,
        type=['pdf'],
        help="Upload PDF documents to enable document-based Q&A",
        key="pdf_uploader"
    )

    # Process uploaded files only if they've changed
    current_hash = get_files_hash(uploaded_files)

    if uploaded_files and current_hash != st.session_state.uploaded_files_hash and not st.session_state.processing_documents:
        st.session_state.processing_documents = True
        st.session_state.uploaded_files_hash = current_hash

        upload_path = os.getenv('UPLOAD_PATH', './data/uploads')
        os.makedirs(upload_path, exist_ok=True)

        # Save files to disk
        saved_files = []
        new_files = []

        with st.spinner("📁 Saving files..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_path, uploaded_file.name)

                file_exists = os.path.exists(file_path)
                if file_exists:
                    with open(file_path, 'rb') as f:
                        existing_content = f.read()
                    new_content = bytes(uploaded_file.getbuffer())
                    if existing_content == new_content:
                        st.info(f"📄 {uploaded_file.name} already exists")
                        continue

                with open(file_path, "wb") as f:
                    f.write(bytes(uploaded_file.getbuffer()))
                saved_files.append(uploaded_file.name)
                new_files.append(uploaded_file.name)
                st.success(f"✅ Saved: {uploaded_file.name}")

        # Index documents if there are new files
        if new_files:
            with st.spinner(f"📚 Indexing {len(new_files)} new documents..."):
                success = st.session_state.agent.load_documents(force_reload=False)
                if success:
                    st.success(f"✅ Successfully indexed {len(new_files)} new documents!")
                else:
                    st.error("❌ Failed to index some documents")

        st.session_state.processing_documents = False

        # Clear the file uploader by rerunning
        time.sleep(1)
        st.rerun()

    # System Info
    st.markdown("---")
    st.header("📊 System Status")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", system_info['documents_loaded'])
    with col2:
        st.metric("Chat History", system_info['conversation_length'])

    # Index status with more detail
    if system_info['index_ready']:
        st.success(f"✅ Index Ready")
        st.caption(f"📁 {system_info.get('indexed_files_count', 0)} files indexed")
        st.caption(f"🗄️ {system_info.get('collection_count', 0)} vectors in DB")
    else:
        st.warning("⚠️ No documents indexed")
        st.caption("Upload PDFs to enable document search")

    # Document list
    if system_info['document_list']:
        with st.expander(f"📁 Loaded Documents ({len(system_info['document_list'])})", expanded=False):
            for doc in system_info['document_list']:
                display_name = doc if len(doc) <= 35 else doc[:32] + "..."
                st.text(f"📄 {display_name}")

    # Tool Status
    st.markdown("---")
    st.header("🛠️ Available Tools")

    tools_status = {
        "📚 RAG Search": "✅ Ready" if system_info['index_ready'] else "⚠️ No documents",
        "🌐 Web Search": "✅ Ready",
        "🤖 Claude AI": "✅ Connected",
        "🗄️ Vector DB": "✅ ChromaDB"
    }

    for tool, status in tools_status.items():
        st.write(f"{tool}: {status}")

    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.clear_conversation()
            st.rerun()

    with col2:
        if st.button("🔄 Re-index All", use_container_width=True):
            with st.spinner("🔄 Re-indexing all documents..."):
                success = st.session_state.agent.load_documents(force_reload=True)
                if success:
                    st.success("✅ Re-indexed successfully!")
                else:
                    st.error("❌ Re-indexing failed")
            time.sleep(1)
            st.rerun()

    # About section
    st.markdown("---")
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        ### Intelligent RAG Assistant

        **Features:**
        - 📚 **Document RAG**: Query your PDFs
        - 🌐 **Web Search**: Current information
        - 🤖 **Claude AI**: Smart responses
        - 🔄 **Auto-fallback**: Web search when needed
        - 💾 **Persistent**: Documents stay indexed

        **How it works:**
        1. Upload PDFs for context
        2. Ask any question
        3. Gets answers from docs or web

        **Tips:**
        - Documents are indexed once and persist
        - System auto-searches web if needed
        - Combines sources for best answers
        """)

# Main chat interface
st.title("💬 Intelligent Q&A Assistant")

# Add a subtitle with current capabilities
system_info = st.session_state.agent.get_system_info()
if system_info['index_ready']:
    st.caption(f"📚 {system_info['documents_loaded']} documents ready | 🌐 Web search enabled | 🤖 Claude AI connected")
else:
    st.caption("🌐 Web search enabled | 🤖 Claude AI ready | 📄 Upload PDFs to enable document search")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display metadata for assistant messages
        if message["role"] == "assistant":
            metadata_cols = st.columns(4)

            with metadata_cols[0]:
                if "tool_used" in message:
                    tool_emoji = {
                        "rag_search": "📚",
                        "web_search": "🌐",
                        "general_knowledge": "🧠",
                        "rag_search + web_search": "📚🌐",
                        "combined": "📚🌐"
                    }.get(message['tool_used'], "🛠️")
                    st.caption(f"{tool_emoji} {message['tool_used']}")

            with metadata_cols[1]:
                if "confidence" in message:
                    confidence = message.get('confidence', 0)
                    if confidence >= 0.8:
                        st.caption(f"✅ High: {confidence:.0%}")
                    elif confidence >= 0.6:
                        st.caption(f"📊 Medium: {confidence:.0%}")
                    else:
                        st.caption(f"⚠️ Low: {confidence:.0%}")

            with metadata_cols[2]:
                if "timestamp" in message:
                    st.caption(f"⏱️ {message.get('timestamp', '')}")

            with metadata_cols[3]:
                if "sources" in message and message["sources"]:
                    with st.popover("📚 Sources"):
                        for source in message["sources"]:
                            if isinstance(source, str) and source.startswith("http"):
                                display_text = source[:50] + "..." if len(source) > 50 else source
                                st.markdown(f"🔗 [{display_text}]({source})")
                            else:
                                st.markdown(f"📄 {source}")

# Chat input
if prompt := st.chat_input("Ask anything - I'll search documents and the web as needed..."):
    if st.session_state.processing_documents:
        st.warning("⏳ Please wait, documents are being indexed...")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            metadata_placeholder = st.empty()

            with st.status("🤔 Processing your query...", expanded=True) as status:
                st.write("🔍 Analyzing query...")
                if st.session_state.agent.index is not None:
                    st.write("📚 Searching documents...")
                st.write("🌐 Determining if web search needed...")

                response: AgentResponse = st.session_state.agent.process_query(prompt)

                status.update(label="✅ Complete!", state="complete", expanded=False)

            message_placeholder.markdown(response.answer)

            with metadata_placeholder.container():
                metadata_cols = st.columns(4)

                with metadata_cols[0]:
                    tool_emoji = {
                        "rag_search": "📚",
                        "web_search": "🌐",
                        "general_knowledge": "🧠",
                        "rag_search + web_search": "📚🌐",
                        "combined": "📚🌐"
                    }.get(response.tool_used, "🛠️")
                    st.caption(f"{tool_emoji} {response.tool_used}")

                with metadata_cols[1]:
                    if response.confidence >= 0.8:
                        st.caption(f"✅ High: {response.confidence:.0%}")
                    elif response.confidence >= 0.6:
                        st.caption(f"📊 Medium: {response.confidence:.0%}")
                    else:
                        st.caption(f"⚠️ Low: {response.confidence:.0%}")

                with metadata_cols[2]:
                    st.caption(f"⏱️ {datetime.now().strftime('%H:%M:%S')}")

                with metadata_cols[3]:
                    if response.sources:
                        with st.popover("📚 Sources"):
                            for source in response.sources:
                                if isinstance(source, str) and source.startswith("http"):
                                    display_text = source[:50] + "..." if len(source) > 50 else source
                                    st.markdown(f"🔗 [{display_text}]({source})")
                                else:
                                    st.markdown(f"📄 {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "tool_used": response.tool_used,
                "confidence": response.confidence,
                "timestamp": datetime.now().strftime('%H:%M:%S')
            })

# Helpful prompts if no messages
if not st.session_state.messages:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📚 Document Questions")
        st.markdown("""
        *After uploading PDFs:*
        - "Summarize the main points"
        - "What does it say about [topic]?"
        - "Find information about [term]"
        """)

    with col2:
        st.markdown("### 🌐 Web Search")
        st.markdown("""
        *Current information:*
        - "Latest news about AI"
        - "Current trends in technology"
        - "Recent developments in [topic]"
        """)

    with col3:
        st.markdown("### 🧠 General Knowledge")
        st.markdown("""
        *Any topic:*
        - "Explain quantum computing"
        - "How does machine learning work?"
        - "Best practices for [topic]"
        """)

    st.info("""
    💡 **Pro tip:** The assistant automatically determines the best source for your answer:
    - Documents first (if relevant PDFs are uploaded)
    - Web search for current/missing information
    - General knowledge for established facts

    Your documents are indexed once and persist between sessions!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by Claude AI 🤖 | Vector Storage by ChromaDB 🗄️ | Web Search by Google 🔎</p>
    <p>Truly Agentic RAG Assistant v2.1 - Documents persist between sessions!</p>
</div>
""", unsafe_allow_html=True)
