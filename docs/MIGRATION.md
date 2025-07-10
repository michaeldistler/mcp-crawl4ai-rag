# Migration Guide: LLM Provider Support

This guide helps you migrate from the previous OpenAI-only version to the new multi-provider system.

## What Changed

### Before (OpenAI Only)
```bash
# Old .env configuration
OPENAI_API_KEY=your_key_here
MODEL_CHOICE=gpt-4o-mini
```

### After (Multi-Provider Support)
```bash
# New .env configuration
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional: Legacy support
MODEL_CHOICE=gpt-4o-mini  # Still works as fallback
```

## Migration Steps

### Step 1: Update Environment Variables

**Minimal migration** (keep using OpenAI):
```bash
# Add these new variables to your .env file
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Keep your existing variables
OPENAI_API_KEY=your_existing_key
MODEL_CHOICE=your_existing_model  # Optional, LLM_MODEL takes precedence
```

**Recommended migration**:
```bash
# Replace MODEL_CHOICE with LLM_MODEL
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
LLM_MODEL=gpt-4o-mini                    # Replaces MODEL_CHOICE
OPENAI_API_KEY=your_existing_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Step 2: Update Dependencies

The new system includes additional optional dependencies:

```bash
# If using Docker Compose, rebuild containers
docker-compose down
docker-compose up --build

# If using uv locally, reinstall dependencies
uv sync
```

### Step 3: Test Your Configuration

Run the provider test script to ensure everything works:

```bash
python test_llm_providers.py
```

### Step 4: (Optional) Switch to Different Providers

Now you can explore other providers:

**Switch to local Ollama**:
```bash
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSION=768
# Remove OPENAI_API_KEY if not using OpenAI
```

**Use Anthropic for completions + local embeddings**:
```bash
LLM_PROVIDER=anthropic
EMBEDDING_PROVIDER=sentence_transformers
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_anthropic_key
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
```

## Breaking Changes

### None for Basic Usage
If you're using the basic OpenAI setup, **no breaking changes** exist. Your existing configuration will continue to work.

### Environment Variable Priority
The system now uses this priority order for model selection:
1. `LLM_MODEL` (new, recommended)
2. `MODEL_CHOICE` (legacy, still supported)
3. Provider defaults

### Embedding Dimensions
When switching embedding providers, the vector dimensions may change:
- OpenAI text-embedding-3-small: 1536 dimensions
- Ollama nomic-embed-text: 768 dimensions  
- Sentence Transformers all-MiniLM-L6-v2: 384 dimensions

**Important**: If you switch embedding providers, you may need to clear existing embeddings and re-crawl content.

## Troubleshooting

### "Provider not available" errors
- Check that API keys are set correctly
- For Ollama, ensure the service is running: `ollama serve`
- For Sentence Transformers, ensure models can download

### Import errors
- Update dependencies: `uv sync` or rebuild Docker containers
- Check that optional dependencies are installed

### Dimension mismatch errors
- Clear existing embeddings when switching providers:
  ```sql
  TRUNCATE TABLE document_chunks, code_examples;
  ```
- Re-crawl your content to generate new embeddings

### Legacy MODEL_CHOICE not working
- Add `LLM_MODEL` to your .env file
- The system prioritizes `LLM_MODEL` over `MODEL_CHOICE`

## Rollback Instructions

If you need to rollback to the previous version:

1. **Keep your environment variables** - the new system is backward compatible
2. **Use git to rollback**:
   ```bash
   git checkout previous_version_tag
   ```
3. **Rebuild if using Docker**:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## Configuration Examples

### Gradual Migration Path

**Week 1**: Add new variables, keep existing setup
```bash
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=existing_key
MODEL_CHOICE=existing_model      # Keep this
LLM_MODEL=gpt-4o-mini           # Add this
```

**Week 2**: Test different embedding provider
```bash
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=sentence_transformers
OPENAI_API_KEY=existing_key
LLM_MODEL=gpt-4o-mini
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
```

**Week 3**: Fully local setup
```bash
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Getting Help

- Review [LLM_PROVIDERS.md](LLM_PROVIDERS.md) for detailed configuration
- Run `python test_llm_providers.py` to test your setup
- Check logs for specific error messages
- Open GitHub issues for migration problems

## Benefits of Migration

After migration, you can:
- Run everything locally with Ollama (complete privacy)
- Use different providers for embeddings vs completions
- Switch providers without code changes
- Access higher-quality models (Claude, GPT-4o)
- Reduce costs with local embedding models
- Improve performance with specialized models
