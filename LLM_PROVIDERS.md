# LLM Provider Configuration Guide

The mcp-crawl4ai-rag system supports multiple LLM providers for both embeddings and completions, allowing you to choose the best provider for your needs or switch between them easily.

## Supported Providers

### LLM Providers (for completions)
- **OpenAI**: GPT models including GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude models including Claude-3.5-Sonnet, Claude-3-Haiku
- **Ollama**: Local models like Llama 3.2, Mistral, CodeLlama (requires local Ollama installation)

### Embedding Providers
- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Ollama**: Local embedding models like nomic-embed-text, mxbai-embed-large
- **Sentence Transformers**: Local embeddings using Hugging Face models (runs offline)

## Configuration

Set the following environment variables in your `.env` file:

### Basic Provider Selection
```bash
# Choose your LLM provider: "openai", "anthropic", or "ollama"
LLM_PROVIDER=openai

# Choose your embedding provider: "openai", "ollama", or "sentence_transformers"
EMBEDDING_PROVIDER=openai

# Default model for your chosen LLM provider
LLM_MODEL=gpt-4o-mini
```

### Provider-Specific Configuration

#### OpenAI Configuration
```bash
# Required for OpenAI LLM and/or embedding services
OPENAI_API_KEY=your_openai_api_key_here

# Embedding model (when using OpenAI embeddings)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

#### Anthropic Configuration
```bash
# Required for Anthropic LLM services
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

#### Ollama Configuration
```bash
# Base URL for your Ollama instance
OLLAMA_BASE_URL=http://localhost:11434

# Embedding model (when using Ollama embeddings)
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Embedding dimension (depends on your chosen model)
OLLAMA_EMBEDDING_DIMENSION=768
```

#### Sentence Transformers Configuration
```bash
# Model name from Hugging Face (when using local embeddings)
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
```

## Example Configurations

### Configuration 1: All OpenAI
Best for cloud-based deployment with high-quality results.
```bash
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Configuration 2: Anthropic + OpenAI Embeddings
High-quality completions with established embeddings.
```bash
LLM_PROVIDER=anthropic
EMBEDDING_PROVIDER=openai
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Configuration 3: Fully Local with Ollama
Complete privacy and offline operation.
```bash
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSION=768
```

### Configuration 4: Local Embeddings + Cloud LLM
Balance between privacy for embeddings and quality for completions.
```bash
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=sentence_transformers
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
```

## Provider-Specific Model Options

### OpenAI Models
- **LLM Models**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4-turbo`
- **Embedding Models**: `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d), `text-embedding-ada-002` (1536d)

### Anthropic Models
- **Claude 3.5**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- **Claude 3**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

### Ollama Models (Local)
First install models with `ollama pull <model_name>`:
- **LLM Models**: `llama3.2`, `llama3.1`, `mistral`, `codellama`, `gemma2`
- **Embedding Models**: `nomic-embed-text`, `mxbai-embed-large`, `snowflake-arctic-embed`

### Sentence Transformers Models
Popular options from Hugging Face:
- `all-MiniLM-L6-v2` (384d) - Fast and lightweight
- `all-mpnet-base-v2` (768d) - Better quality
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

## Setting Up Local Providers

### Ollama Setup
1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the models you want to use:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Start Ollama service:
   ```bash
   ollama serve
   ```

### Sentence Transformers Setup
Models are automatically downloaded when first used. No additional setup required beyond the Python package installation.

## Testing Your Configuration

Use the provided test script to verify your provider setup:

```bash
python test_llm_providers.py
```

This script will:
- Check which providers are available
- Test embedding generation with each provider
- Test completion generation with each provider
- Demonstrate provider switching functionality

## Performance and Cost Considerations

### Speed
1. **Sentence Transformers** (local) - Fastest for embeddings
2. **Ollama** (local) - Fast, depends on hardware
3. **OpenAI** (API) - Medium latency
4. **Anthropic** (API) - Medium latency

### Cost
1. **Sentence Transformers** - Free (local)
2. **Ollama** - Free (local, requires hardware)
3. **OpenAI** - Pay per token
4. **Anthropic** - Pay per token

### Quality
1. **GPT-4o** (OpenAI) - Highest quality completions
2. **Claude-3.5-Sonnet** (Anthropic) - Very high quality completions
3. **text-embedding-3-large** (OpenAI) - Highest quality embeddings
4. **Local models** - Good quality, varies by model size

## Troubleshooting

### Common Issues

#### Provider Not Available
- Check API keys are set correctly
- For Ollama, ensure service is running and models are pulled
- For Sentence Transformers, ensure models can be downloaded

#### Dimension Mismatch Errors
- When switching embedding providers, the vector dimensions may change
- Clear existing embeddings or recreate the database when switching
- Set `OLLAMA_EMBEDDING_DIMENSION` correctly for your chosen model

#### Connection Errors
- For Ollama, check `OLLAMA_BASE_URL` is correct
- For API providers, check internet connectivity and API key validity

### Debugging

Enable debug logging to see detailed provider information:
```bash
export PYTHONPATH=/path/to/your/project/src
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from llm_providers import get_provider_manager
pm = get_provider_manager()
print(pm.list_available_providers())
"
```

## Migration Between Providers

When switching providers, especially embedding providers:

1. **Backup your data** if needed
2. **Update environment variables** with new provider settings
3. **Clear existing embeddings** if dimension changes:
   ```sql
   TRUNCATE TABLE document_chunks, code_examples;
   ```
4. **Re-crawl content** to generate new embeddings
5. **Test functionality** with the new provider

## Advanced Usage

### Dynamic Provider Switching
You can programmatically switch providers in your code:

```python
from llm_providers import get_provider_manager, LLMProvider, EmbeddingProvider

manager = get_provider_manager()

# Use OpenAI for this specific request
completion = manager.create_completion(
    messages=[{"role": "user", "content": "Hello"}],
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini"
)

# Use local embeddings for this request
embedding = manager.create_embedding(
    "test text",
    provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
)
```

### Custom Model Parameters
Each provider supports different parameters:

```python
# OpenAI with custom parameters
completion = manager.create_completion(
    messages=[...],
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000
)

# Anthropic with custom parameters
completion = manager.create_completion(
    messages=[...],
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet-20241022",
    temperature=0.3,
    max_tokens=2000
)
```
