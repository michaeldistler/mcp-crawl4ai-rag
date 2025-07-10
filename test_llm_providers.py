#!/usr/bin/env python3
"""
Test script for LLM provider switching functionality.
Demonstrates how to use different providers for embeddings and completions.
"""
import os
import sys
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_providers import (
    get_provider_manager, 
    LLMProvider, 
    EmbeddingProvider
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_provider_availability():
    """Test which providers are available."""
    provider_manager = get_provider_manager()
    availability = provider_manager.list_available_providers()
    
    print("=== Provider Availability ===")
    print("\nLLM Providers:")
    for provider, available in availability["llm_providers"].items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {provider}: {status}")
    
    print("\nEmbedding Providers:")
    for provider, available in availability["embedding_providers"].items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {provider}: {status}")
    
    return availability


def test_embedding_providers():
    """Test embedding creation with different providers."""
    print("\n=== Testing Embedding Providers ===")
    provider_manager = get_provider_manager()
    test_text = "This is a test sentence for embedding generation."
    
    for provider in EmbeddingProvider:
        try:
            embedding_provider = provider_manager.get_embedding_provider(provider)
            if embedding_provider.is_available():
                print(f"\nTesting {provider.value}...")
                embedding = provider_manager.create_embedding(test_text, provider)
                dimension = provider_manager.get_embedding_dimension(provider)
                print(f"  ✓ Generated embedding with dimension: {dimension}")
                print(f"  ✓ First 5 values: {embedding[:5]}")
            else:
                print(f"\n  ✗ {provider.value} not available")
        except Exception as e:
            print(f"\n  ✗ Error with {provider.value}: {e}")


def test_llm_providers():
    """Test completion generation with different providers."""
    print("\n=== Testing LLM Providers ===")
    provider_manager = get_provider_manager()
    
    test_messages = [
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]
    
    # Provider-specific models
    models = {
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
        LLMProvider.OLLAMA: "llama3.2"  # Assuming this model is available
    }
    
    for provider in LLMProvider:
        try:
            llm_provider = provider_manager.get_llm_provider(provider)
            if llm_provider.is_available():
                print(f"\nTesting {provider.value}...")
                model = models.get(provider, provider_manager.default_llm_model)
                completion = provider_manager.create_completion(
                    test_messages, 
                    model=model,
                    provider=provider,
                    temperature=0.1,
                    max_tokens=10
                )
                print(f"  ✓ Generated completion: '{completion.strip()}'")
            else:
                print(f"\n  ✗ {provider.value} not available")
        except Exception as e:
            print(f"\n  ✗ Error with {provider.value}: {e}")


def test_batch_embeddings():
    """Test batch embedding generation."""
    print("\n=== Testing Batch Embeddings ===")
    provider_manager = get_provider_manager()
    
    test_texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    
    for provider in EmbeddingProvider:
        try:
            embedding_provider = provider_manager.get_embedding_provider(provider)
            if embedding_provider.is_available():
                print(f"\nTesting batch embeddings with {provider.value}...")
                embeddings = provider_manager.create_embeddings_batch(test_texts, provider)
                print(f"  ✓ Generated {len(embeddings)} embeddings")
                if embeddings:
                    dimension = len(embeddings[0])
                    print(f"  ✓ Each embedding has dimension: {dimension}")
            else:
                print(f"\n  ✗ {provider.value} not available")
        except Exception as e:
            print(f"\n  ✗ Error with {provider.value}: {e}")


def demonstrate_provider_switching():
    """Demonstrate switching between providers dynamically."""
    print("\n=== Demonstrating Provider Switching ===")
    
    # Test switching between available embedding providers
    provider_manager = get_provider_manager()
    test_text = "Dynamic provider switching test."
    
    print(f"\nDefault LLM provider: {provider_manager.default_llm_provider.value}")
    print(f"Default embedding provider: {provider_manager.default_embedding_provider.value}")
    
    # Try each embedding provider if available
    available_embedding_providers = []
    for provider in EmbeddingProvider:
        embedding_provider = provider_manager.get_embedding_provider(provider)
        if embedding_provider.is_available():
            available_embedding_providers.append(provider)
    
    if len(available_embedding_providers) > 1:
        print(f"\nSwitching between {len(available_embedding_providers)} embedding providers:")
        for provider in available_embedding_providers:
            try:
                embedding = provider_manager.create_embedding(test_text, provider)
                dimension = len(embedding)
                print(f"  ✓ {provider.value}: dimension {dimension}, first value: {embedding[0]:.4f}")
            except Exception as e:
                print(f"  ✗ {provider.value}: {e}")
    else:
        print("\nOnly one embedding provider available, cannot demonstrate switching.")


def main():
    """Main test function."""
    print("LLM Provider Test Suite")
    print("=" * 50)
    
    # Check environment variables
    print("\nEnvironment Configuration:")
    env_vars = [
        "LLM_PROVIDER", "EMBEDDING_PROVIDER", "LLM_MODEL",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if var.endswith("_KEY") and value:
            value = "***" + value[-4:] if len(value) > 4 else "***"
        print(f"  {var}: {value or 'Not set'}")
    
    try:
        # Run tests
        availability = test_provider_availability()
        
        # Only run provider tests if any are available
        if any(availability["embedding_providers"].values()):
            test_embedding_providers()
            test_batch_embeddings()
        else:
            print("\n⚠️  No embedding providers available. Check your configuration.")
        
        if any(availability["llm_providers"].values()):
            test_llm_providers()
        else:
            print("\n⚠️  No LLM providers available. Check your configuration.")
        
        demonstrate_provider_switching()
        
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
