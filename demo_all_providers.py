#!/usr/bin/env python3
"""
Demo script showing different LLM provider configurations.
"""

import os

def demo_provider_config(provider_name, config):
    """Demo a specific provider configuration."""
    print(f"\n{'=' * 60}")
    print(f"🚀 Demo: {provider_name} Configuration")
    print(f"{'=' * 60}")
    
    # Set environment variables
    for key, value in config.items():
        os.environ[key] = value
    
    # Display LLM Provider Configuration
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", os.getenv("MODEL_CHOICE", "gpt-4o-mini"))
    
    print(f"🧠 LLM Provider: {llm_provider}")
    print(f"📊 Embedding Provider: {embedding_provider}")
    print(f"🤖 LLM Model: {llm_model}")
    
    # Display provider-specific configuration
    if llm_provider == "openai" or embedding_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"🔑 OpenAI API Key: {'✓ Configured' if openai_key else '✗ Missing'}")
        if embedding_provider == "openai":
            print(f"📈 OpenAI Embedding Model: {openai_embedding_model}")
    
    if llm_provider == "anthropic":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        print(f"🔑 Anthropic API Key: {'✓ Configured' if anthropic_key else '✗ Missing'}")
    
    if llm_provider == "ollama" or embedding_provider == "ollama":
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"🏠 Ollama Base URL: {ollama_base_url}")
        if embedding_provider == "ollama":
            ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            ollama_dimension = os.getenv("OLLAMA_EMBEDDING_DIMENSION", "768")
            print(f"📈 Ollama Embedding Model: {ollama_embedding_model}")
            print(f"📏 Ollama Embedding Dimension: {ollama_dimension}")
    
    if embedding_provider == "sentence_transformers":
        st_model = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
        print(f"📈 Sentence Transformers Model: {st_model}")
    
    print()

def main():
    """Demo different provider configurations."""
    
    print("🎯 LLM Provider Configuration Logging Demo")
    print("This demonstrates how the MCP server displays provider configuration on startup")
    
    # OpenAI Configuration
    openai_config = {
        "LLM_PROVIDER": "openai",
        "EMBEDDING_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-demo-key-12345",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
    }
    demo_provider_config("OpenAI", openai_config)
    
    # Anthropic Configuration  
    anthropic_config = {
        "LLM_PROVIDER": "anthropic",
        "EMBEDDING_PROVIDER": "sentence_transformers",
        "LLM_MODEL": "claude-3-5-sonnet-20241022",
        "ANTHROPIC_API_KEY": "sk-ant-demo-key-12345",
        "SENTENCE_TRANSFORMERS_MODEL": "all-MiniLM-L6-v2"
    }
    demo_provider_config("Anthropic + Sentence Transformers", anthropic_config)
    
    # Ollama Configuration
    ollama_config = {
        "LLM_PROVIDER": "ollama", 
        "EMBEDDING_PROVIDER": "ollama",
        "LLM_MODEL": "llama3.2:3b",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
        "OLLAMA_EMBEDDING_DIMENSION": "768"
    }
    demo_provider_config("Ollama (Local)", ollama_config)
    
    # Mixed Configuration
    mixed_config = {
        "LLM_PROVIDER": "openai",
        "EMBEDDING_PROVIDER": "ollama", 
        "LLM_MODEL": "gpt-4o",
        "OPENAI_API_KEY": "sk-demo-key-67890",
        "OLLAMA_BASE_URL": "http://192.168.1.100:11434",
        "OLLAMA_EMBEDDING_MODEL": "mxbai-embed-large",
        "OLLAMA_EMBEDDING_DIMENSION": "1024"
    }
    demo_provider_config("Mixed (OpenAI LLM + Ollama Embeddings)", mixed_config)
    
    print(f"\n{'=' * 60}")
    print("✅ All provider configuration demos complete!")
    print("💡 The actual MCP server would also test provider availability")
    print("   and show warnings for any missing/misconfigured providers.")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
