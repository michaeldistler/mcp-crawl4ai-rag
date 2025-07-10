#!/usr/bin/env python3
"""
Simple validation script for LLM provider structure.
Tests provider configuration and structure without requiring external dependencies.
"""
import os
import sys
from unittest.mock import patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_provider_structure():
    """Test the basic provider structure."""
    print("Testing LLM Provider Structure...")
    
    # Mock external dependencies to avoid import errors
    sys.modules['openai'] = type(sys)('openai')
    sys.modules['anthropic'] = type(sys)('anthropic')
    
    try:
        from llm_providers import (
            LLMProvider, EmbeddingProvider, LLMProviderManager,
            OpenAILLMProvider, AnthropicLLMProvider, OllamaLLMProvider,
            get_provider_manager
        )
        print("✓ All provider classes imported successfully")
        
        # Test enums
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OLLAMA.value == "ollama"
        print("✓ LLM provider enum values correct")
        
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.OLLAMA.value == "ollama"
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS.value == "sentence_transformers"
        print("✓ Embedding provider enum values correct")
        
        # Test provider manager instantiation
        manager = LLMProviderManager()
        assert manager is not None
        print("✓ LLM provider manager instantiated")
        
        # Test default configuration
        assert manager.default_llm_provider == LLMProvider.OPENAI
        assert manager.default_embedding_provider == EmbeddingProvider.OPENAI
        assert manager.default_llm_model == "gpt-4o-mini"
        print("✓ Default configuration correct")
        
        # Test availability checking structure
        availability = manager.list_available_providers()
        assert "llm_providers" in availability
        assert "embedding_providers" in availability
        print("✓ Availability checking structure correct")
        
        # Test provider retrieval
        llm_provider = manager.get_llm_provider()
        embedding_provider = manager.get_embedding_provider()
        assert llm_provider is not None
        assert embedding_provider is not None
        print("✓ Provider retrieval working")
        
        # Test global manager
        global_manager = get_provider_manager()
        assert global_manager is not None
        print("✓ Global provider manager working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing provider structure: {e}")
        return False

def test_environment_configuration():
    """Test environment variable configuration."""
    print("\nTesting Environment Configuration...")
    
    # Mock external dependencies
    sys.modules['openai'] = type(sys)('openai')
    sys.modules['anthropic'] = type(sys)('anthropic')
    
    try:
        from llm_providers import LLMProvider, EmbeddingProvider, LLMProviderManager
        
        # Test default configuration
        with patch.dict(os.environ, {}, clear=True):
            manager = LLMProviderManager()
            assert manager.default_llm_provider == LLMProvider.OPENAI
            assert manager.default_embedding_provider == EmbeddingProvider.OPENAI
            print("✓ Default environment configuration correct")
        
        # Test custom configuration
        custom_env = {
            'LLM_PROVIDER': 'anthropic',
            'EMBEDDING_PROVIDER': 'sentence_transformers',
            'LLM_MODEL': 'claude-3-haiku'
        }
        
        with patch.dict(os.environ, custom_env, clear=True):
            manager = LLMProviderManager()
            assert manager.default_llm_provider == LLMProvider.ANTHROPIC
            assert manager.default_embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS
            assert manager.default_llm_model == 'claude-3-haiku'
            print("✓ Custom environment configuration correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing environment configuration: {e}")
        return False

def test_postgres_utils_integration():
    """Test that postgres_utils can import the provider manager."""
    print("\nTesting PostgreSQL Utils Integration...")
    
    # Mock external dependencies more thoroughly
    sys.modules['openai'] = type(sys)('openai')
    sys.modules['anthropic'] = type(sys)('anthropic')
    
    # Create a mock psycopg2 module with the required classes
    mock_psycopg2 = type(sys)('psycopg2')
    mock_psycopg2_extras = type(sys)('psycopg2_extras')
    mock_psycopg2_extras.RealDictCursor = type('RealDictCursor', (), {})
    mock_psycopg2_extras.execute_batch = lambda *args: None
    mock_psycopg2.connect = lambda *args, **kwargs: None
    
    sys.modules['psycopg2'] = mock_psycopg2
    sys.modules['psycopg2.extras'] = mock_psycopg2_extras
    
    try:
        # This should work now that we've updated postgres_utils.py
        from postgres_utils import PostgresClient
        print("✓ PostgreSQL utils can be imported")
        
        # Test that the provider manager import works
        from postgres_utils import create_embedding, create_embeddings_batch
        print("✓ Provider-based functions can be imported from postgres_utils")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing postgres_utils integration: {e}")
        return False

def main():
    """Run all validation tests."""
    print("LLM Provider System Validation")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_provider_structure,
        test_environment_configuration,
        test_postgres_utils_integration
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All validation tests passed!")
        print("\nThe LLM provider system is properly configured and ready to use.")
        print("\nNext steps:")
        print("1. Set up your environment variables in .env")
        print("2. Install dependencies with: uv sync")
        print("3. Test with a real provider using: python test_llm_providers.py")
    else:
        print("✗ Some validation tests failed!")
        print("Please check the errors above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
