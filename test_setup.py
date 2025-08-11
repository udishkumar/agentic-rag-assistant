#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

print("🔍 Testing setup...")

# Load environment variables
load_dotenv()

# Check API key
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key or api_key == 'your_actual_claude_api_key_here':
    print("❌ API key not set properly in .env file")
    print("   Please add your actual Claude API key")
    sys.exit(1)
else:
    print(f"✅ API key found: {api_key[:10]}...")

# Check directories
dirs = ['./data/uploads', './data/vector_store', './app/agents']
for dir_path in dirs:
    if os.path.exists(dir_path):
        print(f"✅ Directory exists: {dir_path}")
    else:
        print(f"❌ Directory missing: {dir_path}")

# Try imports
try:
    from app.agents.rag_agent import AgenticRAG
    print("✅ RAG agent imports successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")

print("\n🎉 Setup test complete!")
