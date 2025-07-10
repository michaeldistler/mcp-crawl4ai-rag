"""
LLM Provider abstraction for the Crawl4AI MCP server.
Supports OpenAI, Ollama, and Anthropic for both embeddings and completions.
"""
import os
import abc
import time
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum

import openai
import anthropic

# Set up logging
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class BaseLLMProvider(abc.ABC):
    """Base class for LLM providers."""
    
    @abc.abstractmethod
    def create_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Create a completion using the provider's API."""
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass


class BaseEmbeddingProvider(abc.ABC):
    """Base class for embedding providers."""
    
    @abc.abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text."""
        pass
    
    @abc.abstractmethod
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass
    
    @abc.abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this provider."""
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI provider for LLM completions."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
    
    def create_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Create a completion using OpenAI's API."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI is properly configured."""
        return self.api_key is not None


class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic provider for LLM completions."""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
    
    def create_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Create a completion using Anthropic's API."""
        if not self.client:
            raise ValueError("Anthropic API key not configured")
        
        try:
            # Convert OpenAI-style messages to Anthropic format
            system_message = ""
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = self.client.messages.create(
                model=model,
                system=system_message if system_message else "You are a helpful assistant.",
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic completion error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Anthropic is properly configured."""
        return self.api_key is not None


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama provider for LLM completions."""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Import requests here to avoid adding it as a required dependency
        try:
            import requests
            self.requests = requests
        except ImportError:
            logger.warning("requests library not installed, Ollama provider will not work")
            self.requests = None
    
    def create_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Create a completion using Ollama's API."""
        if not self.requests:
            raise ValueError("requests library required for Ollama provider")
        
        try:
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens if max_tokens else -1
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if not self.requests:
            return False
        
        try:
            response = self.requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI provider for embeddings."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        if not texts:
            return []
        
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                    # Fallback to individual embeddings
                    embeddings = []
                    for text in texts:
                        try:
                            embedding = self.create_embedding(text)
                            embeddings.append(embedding)
                        except:
                            # Add zero embedding as fallback
                            embeddings.append([0.0] * self.get_embedding_dimension())
                    return embeddings
    
    def is_available(self) -> bool:
        """Check if OpenAI is properly configured."""
        return self.api_key is not None
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of OpenAI embeddings."""
        if self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        elif self.model == "text-embedding-ada-002":
            return 1536
        else:
            return 1536  # Default fallback


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama provider for embeddings."""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        # Import requests here to avoid adding it as a required dependency
        try:
            import requests
            self.requests = requests
        except ImportError:
            logger.warning("requests library not installed, Ollama embedding provider will not work")
            self.requests = None
    
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text."""
        if not self.requests:
            raise ValueError("requests library required for Ollama provider")
        
        try:
            response = self.requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            try:
                embedding = self.create_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to create embedding for text: {e}")
                # Add zero embedding as fallback
                embeddings.append([0.0] * self.get_embedding_dimension())
        return embeddings
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if not self.requests:
            return False
        
        try:
            response = self.requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of Ollama embeddings."""
        # Most Ollama embedding models use 768 dimensions, but this can vary
        return int(os.getenv("OLLAMA_EMBEDDING_DIMENSION", "768"))


class SentenceTransformersEmbeddingProvider(BaseEmbeddingProvider):
    """Sentence Transformers provider for local embeddings."""
    
    def __init__(self):
        self.model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.SentenceTransformer = SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers library not installed")
            self.SentenceTransformer = None
    
    def _get_model(self):
        """Lazy load the model."""
        if self._model is None and self.SentenceTransformer:
            self._model = self.SentenceTransformer(self.model_name)
        return self._model
    
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text."""
        model = self._get_model()
        if not model:
            raise ValueError("sentence-transformers library not installed")
        
        try:
            embedding = model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Sentence Transformers embedding error: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        model = self._get_model()
        if not model:
            raise ValueError("sentence-transformers library not installed")
        
        try:
            embeddings = model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Sentence Transformers batch embedding error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Sentence Transformers is available."""
        return self.SentenceTransformer is not None
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of Sentence Transformers embeddings."""
        model = self._get_model()
        if model:
            return model.get_sentence_embedding_dimension()
        # Default dimension for all-MiniLM-L6-v2
        return 384


class LLMProviderManager:
    """Manager for LLM and embedding providers."""
    
    def __init__(self):
        # Initialize providers
        self.llm_providers = {
            LLMProvider.OPENAI: OpenAILLMProvider(),
            LLMProvider.ANTHROPIC: AnthropicLLMProvider(),
            LLMProvider.OLLAMA: OllamaLLMProvider(),
        }
        
        self.embedding_providers = {
            EmbeddingProvider.OPENAI: OpenAIEmbeddingProvider(),
            EmbeddingProvider.OLLAMA: OllamaEmbeddingProvider(),
            EmbeddingProvider.SENTENCE_TRANSFORMERS: SentenceTransformersEmbeddingProvider(),
        }
        
        # Set default providers from environment variables
        self.default_llm_provider = LLMProvider(
            os.getenv("LLM_PROVIDER", "ollama").lower()
        )
        self.default_embedding_provider = EmbeddingProvider(
            os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()
        )
        
        # Default models
        self.default_llm_model = os.getenv("LLM_MODEL", "llama3.2:latest")
        
    def get_llm_provider(self, provider: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get an LLM provider instance."""
        provider = provider or self.default_llm_provider
        return self.llm_providers[provider]
    
    def get_embedding_provider(self, provider: Optional[EmbeddingProvider] = None) -> BaseEmbeddingProvider:
        """Get an embedding provider instance."""
        provider = provider or self.default_embedding_provider
        return self.embedding_providers[provider]
    
    def create_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Create a completion using the specified provider and model."""
        llm_provider = self.get_llm_provider(provider)
        model = model or self.default_llm_model
        
        if not llm_provider.is_available():
            raise ValueError(f"LLM provider {provider or self.default_llm_provider} is not available")
        
        return llm_provider.create_completion(messages, model, temperature, max_tokens)
    
    def create_embedding(
        self, 
        text: str, 
        provider: Optional[EmbeddingProvider] = None
    ) -> List[float]:
        """Create an embedding using the specified provider."""
        embedding_provider = self.get_embedding_provider(provider)
        
        if not embedding_provider.is_available():
            raise ValueError(f"Embedding provider {provider or self.default_embedding_provider} is not available")
        
        return embedding_provider.create_embedding(text)
    
    def create_embeddings_batch(
        self, 
        texts: List[str], 
        provider: Optional[EmbeddingProvider] = None
    ) -> List[List[float]]:
        """Create embeddings for multiple texts using the specified provider."""
        embedding_provider = self.get_embedding_provider(provider)
        
        if not embedding_provider.is_available():
            raise ValueError(f"Embedding provider {provider or self.default_embedding_provider} is not available")
        
        return embedding_provider.create_embeddings_batch(texts)
    
    def get_embedding_dimension(self, provider: Optional[EmbeddingProvider] = None) -> int:
        """Get the embedding dimension for the specified provider."""
        embedding_provider = self.get_embedding_provider(provider)
        return embedding_provider.get_embedding_dimension()
    
    def list_available_providers(self) -> Dict[str, Dict[str, bool]]:
        """List all providers and their availability status."""
        return {
            "llm_providers": {
                provider.value: self.llm_providers[provider].is_available()
                for provider in LLMProvider
            },
            "embedding_providers": {
                provider.value: self.embedding_providers[provider].is_available()
                for provider in EmbeddingProvider
            }
        }


# Global instance
_provider_manager = None

def get_provider_manager() -> LLMProviderManager:
    """Get the global provider manager instance."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = LLMProviderManager()
    return _provider_manager
