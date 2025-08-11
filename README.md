# 🤖 Agentic RAG Assistant

An intelligent document Q&A system with web search capabilities, powered by Claude AI.

## Features

- 📄 **PDF Document Processing**: Upload and query PDF documents
- 🌐 **Web Search Integration**: Search the web for current information
- �� **Claude AI Integration**: Intelligent responses using Anthropic's Claude
- 💬 **Conversation Memory**: Maintains context across interactions
- 🛠️ **Agentic Decision Making**: Automatically chooses the best tool for each query
- 🗄️ **Vector Storage**: ChromaDB for efficient document retrieval

## Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Claude 3 Sonnet (Anthropic)
- **Vector Database**: ChromaDB
- **Document Processing**: LlamaIndex
- **Web Search**: DuckDuckGo API
- **Embeddings**: Sentence Transformers

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Run: `streamlit run app.py`

## Usage

1. Upload PDF documents using the sidebar
2. Ask questions about the documents or general topics
3. The system will automatically choose between document search and web search
4. View sources and confidence scores for each response
