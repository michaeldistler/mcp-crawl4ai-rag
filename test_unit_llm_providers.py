#!/usr/bin/env python3
"""
Unit tests for the LLM provider system.
Tests provider configuration, initialization, and availability checking.
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_providers import (
    LLMProviderManager,
    LLMProvider,
    EmbeddingProvider,
    OpenAILLMProvider,
    AnthropicLLMProvider,
    OllamaLLMProvider,
    OpenAIEmbeddingProvider,
    OllamaEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
    get_provider_manager
)


class TestLLMProviders(unittest.TestCase):
    """Test LLM provider functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear environment variables to ensure clean test state
        env_vars_to_clear = [
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'LLM_PROVIDER',
            'EMBEDDING_PROVIDER', 'LLM_MODEL', 'OLLAMA_BASE_URL'
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def test_provider_enums(self):
        """Test that provider enums are properly defined."""
        # Test LLM provider enum
        self.assertEqual(LLMProvider.OPENAI.value, "openai")
        self.assertEqual(LLMProvider.ANTHROPIC.value, "anthropic")
        self.assertEqual(LLMProvider.OLLAMA.value, "ollama")
        
        # Test embedding provider enum
        self.assertEqual(EmbeddingProvider.OPENAI.value, "openai")
        self.assertEqual(EmbeddingProvider.OLLAMA.value, "ollama")
        self.assertEqual(EmbeddingProvider.SENTENCE_TRANSFORMERS.value, "sentence_transformers")
    
    def test_openai_llm_provider_without_key(self):
        """Test OpenAI LLM provider without API key."""
        provider = OpenAILLMProvider()
        self.assertFalse(provider.is_available())
        self.assertIsNone(provider.api_key)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_openai_llm_provider_with_key(self):
        """Test OpenAI LLM provider with API key."""
        provider = OpenAILLMProvider()
        self.assertTrue(provider.is_available())
        self.assertEqual(provider.api_key, 'test-key')
    
    def test_anthropic_llm_provider_without_key(self):
        """Test Anthropic LLM provider without API key."""
        provider = AnthropicLLMProvider()
        self.assertFalse(provider.is_available())
        self.assertIsNone(provider.api_key)
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    def test_anthropic_llm_provider_with_key(self):
        """Test Anthropic LLM provider with API key."""
        provider = AnthropicLLMProvider()
        self.assertTrue(provider.is_available())
        self.assertEqual(provider.api_key, 'test-key')
    
    @patch('llm_providers.requests')
    def test_ollama_llm_provider_without_requests(self, mock_requests):
        """Test Ollama LLM provider without requests library."""
        mock_requests = None
        provider = OllamaLLMProvider()
        provider.requests = None
        self.assertFalse(provider.is_available())
    
    def test_openai_embedding_provider_without_key(self):
        """Test OpenAI embedding provider without API key."""
        provider = OpenAIEmbeddingProvider()
        self.assertFalse(provider.is_available())
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_openai_embedding_provider_with_key(self):
        """Test OpenAI embedding provider with API key."""
        provider = OpenAIEmbeddingProvider()
        self.assertTrue(provider.is_available())
    
    def test_openai_embedding_dimensions(self):
        """Test OpenAI embedding dimension calculation."""
        provider = OpenAIEmbeddingProvider()
        
        # Test different models
        provider.model = "text-embedding-3-small"
        self.assertEqual(provider.get_embedding_dimension(), 1536)
        
        provider.model = "text-embedding-3-large"
        self.assertEqual(provider.get_embedding_dimension(), 3072)
        
        provider.model = "text-embedding-ada-002"
        self.assertEqual(provider.get_embedding_dimension(), 1536)
        
        provider.model = "unknown-model"
        self.assertEqual(provider.get_embedding_dimension(), 1536)  # Default fallback
    
    @patch('llm_providers.requests')
    def test_ollama_embedding_provider_without_requests(self, mock_requests):
        """Test Ollama embedding provider without requests library."""
        mock_requests = None
        provider = OllamaEmbeddingProvider()
        provider.requests = None
        self.assertFalse(provider.is_available())
    
    @patch.dict(os.environ, {'OLLAMA_EMBEDDING_DIMENSION': '512'})
    def test_ollama_embedding_dimensions(self):
        """Test Ollama embedding dimension from environment."""
        provider = OllamaEmbeddingProvider()
        self.assertEqual(provider.get_embedding_dimension(), 512)
    
    def test_sentence_transformers_provider_without_library(self):
        """Test Sentence Transformers provider without library."""
        provider = SentenceTransformersEmbeddingProvider()
        provider.SentenceTransformer = None
        self.assertFalse(provider.is_available())
    
    @patch.dict(os.environ, {})
    def test_provider_manager_defaults(self):
        """Test provider manager with default configuration."""
        manager = LLMProviderManager()
        
        # Test default providers
        self.assertEqual(manager.default_llm_provider, LLMProvider.OPENAI)
        self.assertEqual(manager.default_embedding_provider, EmbeddingProvider.OPENAI)
        self.assertEqual(manager.default_llm_model, "gpt-4o-mini")
    
    @patch.dict(os.environ, {
        'LLM_PROVIDER': 'anthropic',
        'EMBEDDING_PROVIDER': 'sentence_transformers',
        'LLM_MODEL': 'claude-3-haiku'
    })
    def test_provider_manager_custom_config(self):
        """Test provider manager with custom configuration."""
        manager = LLMProviderManager()
        
        # Test custom providers
        self.assertEqual(manager.default_llm_provider, LLMProvider.ANTHROPIC)
        self.assertEqual(manager.default_embedding_provider, EmbeddingProvider.SENTENCE_TRANSFORMERS)
        self.assertEqual(manager.default_llm_model, "claude-3-haiku")
    
    def test_provider_manager_availability_check(self):
        """Test provider availability checking."""
        manager = LLMProviderManager()
        availability = manager.list_available_providers()
        
        # Should return a dictionary with both provider types
        self.assertIn("llm_providers", availability)
        self.assertIn("embedding_providers", availability)
        
        # Should contain all provider types
        llm_providers = availability["llm_providers"]
        embedding_providers = availability["embedding_providers"]
        
        self.assertIn("openai", llm_providers)
        self.assertIn("anthropic", llm_providers)
        self.assertIn("ollama", llm_providers)
        
        self.assertIn("openai", embedding_providers)
        self.assertIn("ollama", embedding_providers)
        self.assertIn("sentence_transformers", embedding_providers)
        
        # All should be boolean values
        for provider_dict in [llm_providers, embedding_providers]:
            for available in provider_dict.values():
                self.assertIsInstance(available, bool)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('llm_providers.openai.OpenAI')
    def test_create_completion_with_provider(self, mock_openai_client):
        """Test completion creation with specific provider."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client
        
        manager = LLMProviderManager()
        
        messages = [{"role": "user", "content": "Test message"}]
        response = manager.create_completion(
            messages, 
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini"
        )
        
        self.assertEqual(response, "Test response")
        mock_client.chat.completions.create.assert_called_once()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('llm_providers.openai.OpenAI')
    def test_create_embedding_with_provider(self, mock_openai_client):
        """Test embedding creation with specific provider."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_client.return_value = mock_client
        
        manager = LLMProviderManager()
        
        embedding = manager.create_embedding(
            "Test text",
            provider=EmbeddingProvider.OPENAI
        )
        
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        mock_client.embeddings.create.assert_called_once()
    
    def test_global_provider_manager(self):
        """Test global provider manager singleton."""
        manager1 = get_provider_manager()
        manager2 = get_provider_manager()
        
        # Should return the same instance
        self.assertIs(manager1, manager2)
    
    def test_provider_error_handling(self):
        """Test error handling when providers are not available."""
        manager = LLMProviderManager()
        
        # Test completion with unavailable provider
        messages = [{"role": "user", "content": "Test"}]
        
        with self.assertRaises(ValueError) as context:
            manager.create_completion(messages, provider=LLMProvider.OPENAI)
        
        self.assertIn("not available", str(context.exception))
        
        # Test embedding with unavailable provider
        with self.assertRaises(ValueError) as context:
            manager.create_embedding("test", provider=EmbeddingProvider.OPENAI)
        
        self.assertIn("not available", str(context.exception))


class TestProviderIntegration(unittest.TestCase):
    """Integration tests for provider switching."""
    
    def test_provider_switching_configuration(self):
        """Test that provider switching works as expected."""
        # Test different provider combinations
        test_configs = [
            {
                'LLM_PROVIDER': 'openai',
                'EMBEDDING_PROVIDER': 'openai',
                'expected_llm': LLMProvider.OPENAI,
                'expected_embedding': EmbeddingProvider.OPENAI
            },
            {
                'LLM_PROVIDER': 'anthropic',
                'EMBEDDING_PROVIDER': 'sentence_transformers',
                'expected_llm': LLMProvider.ANTHROPIC,
                'expected_embedding': EmbeddingProvider.SENTENCE_TRANSFORMERS
            },
            {
                'LLM_PROVIDER': 'ollama',
                'EMBEDDING_PROVIDER': 'ollama',
                'expected_llm': LLMProvider.OLLAMA,
                'expected_embedding': EmbeddingProvider.OLLAMA
            }
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Create a clean environment dict with only string values
                env_config = {
                    'LLM_PROVIDER': config['LLM_PROVIDER'],
                    'EMBEDDING_PROVIDER': config['EMBEDDING_PROVIDER']
                }
                with patch.dict(os.environ, env_config):
                    manager = LLMProviderManager()
                    
                    self.assertEqual(manager.default_llm_provider, config['expected_llm'])
                    self.assertEqual(manager.default_embedding_provider, config['expected_embedding'])


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
