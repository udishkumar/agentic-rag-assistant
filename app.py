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

# PRAGYA Branding
PROJECT_NAME = "PRAGYA"
PROJECT_TAGLINE = "Persistent Retrieval Augmented Generation Your Assistant"
PROJECT_HINDI = "प्रज्ञा - Your Intelligent Knowledge Companion"
PROJECT_VERSION = "v2.2"
DEVELOPER = "Udish Kumar"

# Page configuration
st.set_page_config(
    page_title=f"{PROJECT_NAME} - Intelligent RAG Assistant",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS (only for special branding elements)
st.markdown("""
<style>
    /* PRAGYA branding gradient */
    .pragya-header {
        background: linear-gradient(90deg, #ff9933, #ffffff, #138808);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 2em;
        text-align: center;
        padding: 10px;
    }
    
    /* Hide Streamlit's hamburger menu if desired */
    #MainMenu {visibility: hidden;}
    
    /* Hide footer if desired */
    footer {visibility: hidden;}
    
    /* Hide the status widget that appears during indexing */
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    
    /* Conversation list styling */
    .conversation-item {
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .conversation-item:hover {
        background-color: rgba(255, 255, 255, 0.1);
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

# Sidebar
with st.sidebar:
    # PRAGYA Branding with gradient
    st.markdown('<p class="pragya-header">🧘 PRAGYA</p>', unsafe_allow_html=True)
    st.caption(PROJECT_HINDI)
    st.markdown("---")

    # Conversation Management Section
    st.header("💬 Conversations")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            create_new_conversation()
    
    with col2:
        if st.button("📋 History", use_container_width=True):
            st.session_state.show_conversation_list = not st.session_state.show_conversation_list
    
    # Show conversation list if toggled
    if st.session_state.show_conversation_list:
        conversations = st.session_state.agent.conversation_manager.list_conversations()
        
        if conversations:
            st.caption(f"📂 {len(conversations)} saved conversations")
            
            # Create a scrollable area for conversations
            with st.container():
                for conv in conversations[:10]:  # Show latest 10
                    conv_date = datetime.fromisoformat(conv['updated_at']).strftime('%m/%d %H:%M')
                    is_current = conv['id'] == st.session_state.current_conversation_id
                    
                    col1, col2, col3 = st.columns([5, 2, 1])
                    
                    with col1:
                        if is_current:
                            st.markdown(f"**✓ {conv['title'][:20]}...**")
                        else:
                            if st.button(f"{conv['title'][:25]}...", key=f"conv_{conv['id']}", use_container_width=True):
                                switch_conversation(conv['id'])
                    
                    with col2:
                        st.caption(conv_date)
                    
                    with col3:
                        if st.button("🗑️", key=f"del_{conv['id']}"):
                            delete_conversation(conv['id'])
        else:
            st.info("No saved conversations yet")
    
    # Current conversation info
    if st.session_state.agent.current_conversation:
        current_conv = st.session_state.agent.current_conversation
        with st.expander("📍 Current Chat", expanded=False):
            st.text(f"Title: {current_conv.title[:30]}")
            st.text(f"Messages: {len(current_conv.messages)}")
            st.text(f"Created: {datetime.fromisoformat(current_conv.created_at).strftime('%Y-%m-%d %H:%M')}")
            
            # Rename conversation
            new_title = st.text_input("Rename:", value=current_conv.title, key="rename_conv")
            if new_title != current_conv.title:
                st.session_state.agent.conversation_manager.update_conversation_title(current_conv.id, new_title)
                st.success("✅ Renamed")

    st.markdown("---")

    # File upload section
    st.header("📄 Documents")

    # Get current system info for display
    system_info = st.session_state.agent.get_system_info()

    # Show current document status
    if system_info['documents_loaded'] > 0:
        st.success(f"✅ {system_info['documents_loaded']} docs ready")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True,
        type=['pdf'],
        help="Upload PDFs for Q&A",
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

        with st.spinner("Saving..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_path, uploaded_file.name)

                file_exists = os.path.exists(file_path)
                if file_exists:
                    with open(file_path, 'rb') as f:
                        existing_content = f.read()
                    new_content = bytes(uploaded_file.getbuffer())
                    if existing_content == new_content:
                        short_name = uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 23 else uploaded_file.name
                        st.info(f"📄 {short_name} exists")
                        continue

                with open(file_path, "wb") as f:
                    f.write(bytes(uploaded_file.getbuffer()))
                saved_files.append(uploaded_file.name)
                new_files.append(uploaded_file.name)
                short_name = uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 23 else uploaded_file.name
                st.success(f"✅ {short_name}")

        # Index documents if there are new files
        if new_files:
            progress_bar = st.progress(0, text="Starting indexing...")
            
            def progress_callback(progress, message):
                progress_bar.progress(progress / 100, text=message)
            
            success = st.session_state.agent.load_documents(
                force_reload=False,
                progress_callback=progress_callback
            )
            
            if success:
                st.success(f"✅ Indexed {len(new_files)} docs!")
            else:
                st.error("❌ Indexing failed")

        st.session_state.processing_documents = False
        time.sleep(1)
        st.rerun()

    # System Info
    st.markdown("---")
    st.header("📊 Status")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Docs", system_info['documents_loaded'])
    with col2:
        st.metric("History", system_info['conversation_length'])

    # Index status
    if system_info['index_ready']:
        st.success("✅ Ready")
        with st.expander("Details"):
            st.caption(f"📁 {system_info.get('indexed_files_count', 0)} files")
            st.caption(f"🗄️ {system_info.get('collection_count', 0)} vectors")
    else:
        st.warning("⚠️ No docs")
        st.caption("Upload PDFs first")

    # Document list
    if system_info['document_list']:
        with st.expander(f"📁 Docs ({len(system_info['document_list'])})", expanded=False):
            for doc in system_info['document_list']:
                display_name = doc if len(doc) <= 25 else doc[:22] + "..."
                st.text(f"📄 {display_name}")

    # Tool Status
    st.markdown("---")
    with st.expander("🛠️ Tools", expanded=False):
        tools_status = {
            "📚 RAG": "✅" if system_info['index_ready'] else "⚠️",
            "🌐 Web": "✅",
            "🤖 AI": "✅",
            "🗄️ DB": "✅"
        }
        
        cols = st.columns(4)
        for i, (tool, status) in enumerate(tools_status.items()):
            with cols[i]:
                st.write(f"{tool}\n{status}")

    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.clear_conversation()
            st.rerun()

    with col2:
        if st.button("🔄 Re-index", use_container_width=True):
            progress_bar = st.progress(0, text="Starting re-indexing...")
            
            def progress_callback(progress, message):
                progress_bar.progress(progress / 100, text=message)
            
            success = st.session_state.agent.load_documents(
                force_reload=True,
                progress_callback=progress_callback
            )
            
            if success:
                st.success("✅ Done!")
            else:
                st.error("❌ Failed")
            time.sleep(1)
            st.rerun()

    # About section
    st.markdown("---")
    with st.expander("ℹ️ About PRAGYA", expanded=False):
        st.markdown(f"""
        **{PROJECT_NAME}** - प्रज्ञा  
        *{PROJECT_TAGLINE}*
        
        **Meaning:** Wisdom, Intelligence, Understanding  
        
        **Features:**  
        📚 Document RAG - Search your PDFs  
        🌐 Web Search - Current information  
        🤖 Claude AI - Intelligent responses  
        🔄 Auto-routing - Best source selection  
        💾 Persistent - Knowledge & conversations preserved  
        💬 Multi-chat - Manage multiple conversations  
        
        **Developer:** {DEVELOPER}  
        **Version:** {PROJECT_VERSION}
        """)

# Main chat interface
st.markdown('<p class="pragya-header">🧘 PRAGYA</p>', unsafe_allow_html=True)
st.caption(f"*{PROJECT_TAGLINE}*")

# Capability display
system_info = st.session_state.agent.get_system_info()
if system_info['index_ready']:
    st.caption(f"📚 {system_info['documents_loaded']} docs | 🌐 Web | 🤖 AI | 💬 Conversation: {st.session_state.agent.current_conversation.title[:30] if st.session_state.agent.current_conversation else 'New'}")
else:
    st.caption(f"🌐 Web | 🤖 AI | 📄 Upload PDFs to unlock full wisdom | 💬 {st.session_state.agent.current_conversation.title[:30] if st.session_state.agent.current_conversation else 'New'}")

# Display chat messages
for message in st.session_state.messages:
    role = message.get("role")
    
    # Handle both old format (user/assistant keys) and new format (role key)
    if role:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(message.get("content", message.get("user", "")))
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message.get("content", message.get("assistant", "")))
                
                # Display metadata
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
                                    display_text = source.split('/')[2] if len(source.split('/')) > 2 else source[:30]
                                    st.markdown(f"🔗 [{display_text}]({source})")
                                else:
                                    display_text = source[:30] + "..." if len(source) > 33 else source
                                    st.markdown(f"📄 {display_text}")
    else:
        # Handle old format
        if "user" in message:
            with st.chat_message("user"):
                st.markdown(message["user"])
        
        if "assistant" in message:
            with st.chat_message("assistant"):
                st.markdown(message["assistant"])
                
                # Display metadata
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
                        try:
                            # Try to parse ISO format first
                            dt = datetime.fromisoformat(message.get('timestamp', ''))
                            st.caption(f"⏱️ {dt.strftime('%H:%M:%S')}")
                        except:
                            # Fall back to raw timestamp
                            st.caption(f"⏱️ {message.get('timestamp', '')}")
                
                with metadata_cols[3]:
                    if "sources" in message and message["sources"]:
                        with st.popover("📚 Sources"):
                            for source in message["sources"]:
                                if isinstance(source, str) and source.startswith("http"):
                                    display_text = source.split('/')[2] if len(source.split('/')) > 2 else source[:30]
                                    st.markdown(f"🔗 [{display_text}]({source})")
                                else:
                                    display_text = source[:30] + "..." if len(source) > 33 else source
                                    st.markdown(f"📄 {display_text}")

# Chat input
if prompt := st.chat_input(f"Ask {PROJECT_NAME} anything... (प्रज्ञा से कुछ भी पूछें)"):
    if st.session_state.processing_documents:
        st.warning("⏳ Indexing in progress...")
    else:
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            metadata_placeholder = st.empty()

            with st.status(f"🧘 {PROJECT_NAME} is thinking...", expanded=False) as status:
                response: AgentResponse = st.session_state.agent.process_query(prompt)
                status.update(label="✅ Wisdom delivered!", state="complete", expanded=False)

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
                                    display_text = source.split('/')[2] if len(source.split('/')) > 2 else source[:30]
                                    st.markdown(f"🔗 [{display_text}]({source})")
                                else:
                                    display_text = source[:30] + "..." if len(source) > 33 else source
                                    st.markdown(f"📄 {display_text}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "tool_used": response.tool_used,
                "confidence": response.confidence,
                "timestamp": datetime.now().isoformat()
            })

# Helpful prompts if no messages
if not st.session_state.messages:
    st.markdown("---")
    
    st.markdown(f"### 💡 Ask {PROJECT_NAME}:")
    
    tab1, tab2, tab3 = st.tabs(["📚 Documents", "🌐 Web", "🧠 General"])
    
    with tab1:
        st.markdown("""
        - Summarize the main points
        - What does it say about X?
        - Find info about Y
        """)

    with tab2:
        st.markdown("""
        - Latest news about AI
        - Current tech trends
        - Recent developments
        """)

    with tab3:
        st.markdown("""
        - Explain quantum computing
        - How does ML work?
        - Best practices for X
        """)

    st.info(f"💡 {PROJECT_NAME} (प्रज्ञा) automatically chooses the best source for your answer - combining ancient wisdom with modern AI!")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 12px; padding: 10px;'>
    <p><strong>{PROJECT_NAME}</strong> - {PROJECT_HINDI}</p>
    <p>Powered by Claude AI 🤖 | ChromaDB 🗄️ | Google 🔎</p>
    <p>Developed with ❤️ by {DEVELOPER} | {PROJECT_VERSION}</p>
</div>
""", unsafe_allow_html=True)