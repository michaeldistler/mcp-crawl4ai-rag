# Docker Compose Setup Guide

This guide explains how to use Docker Compose to run the mcp-crawl4ai-rag project with all its dependencies.

## Prerequisites

- Docker and Docker Compose installed
- `.env` file configured (copy from `.env.example`)

## Quick Start

### 1. Basic Setup (MCP Server + PostgreSQL)

```bash
# Copy and configure environment file
cp .env.example .env
# Edit .env with your configuration

# Start the basic stack
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector extension
- MCP server

### 2. Full Setup (with Knowledge Graph)

```bash
# Enable knowledge graph functionality in .env
echo "USE_KNOWLEDGE_GRAPH=true" >> .env
echo "NEO4J_PASSWORD=your_neo4j_password" >> .env

# Start with knowledge graph profile
docker-compose --profile knowledge-graph up -d
```

This starts:
- PostgreSQL with pgvector extension
- MCP server
- Neo4j database

### 3. Development Setup

```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

This starts everything with development optimizations:
- Volume mounts for live code reloading
- All RAG strategies enabled
- pgAdmin for database management
- Neo4j with development plugins

### 4. Admin Tools

```bash
# Start with admin tools (pgAdmin)
docker-compose --profile admin up -d
```

## Services Overview

### Core Services

#### `postgres`
- **Image**: `pgvector/pgvector:pg16`
- **Purpose**: PostgreSQL database with pgvector extension for vector storage
- **Port**: `5432`
- **Database**: `crawl4ai_rag`
- **Initialization**: Automatically runs `crawled_pages.sql` on first start

#### `mcp-server`
- **Build**: From local Dockerfile
- **Purpose**: Main MCP server for web crawling and RAG
- **Port**: `8051` (configurable via `PORT` env var)
- **Dependencies**: PostgreSQL (required), Neo4j (optional)

### Optional Services

#### `neo4j` (profile: `knowledge-graph`)
- **Image**: `neo4j:5.15-community`
- **Purpose**: Knowledge graph database for AI hallucination detection
- **Ports**: `7474` (browser), `7687` (bolt)
- **Plugins**: APOC included

#### `pgadmin` (profile: `admin`)
- **Image**: `dpage/pgadmin4:latest`
- **Purpose**: PostgreSQL administration interface
- **Port**: `5050`
- **Login**: `admin@crawl4ai.com` / configured password

## Environment Configuration

### Required Variables

```bash
# OpenAI API key for embeddings
OPENAI_API_KEY=your_openai_api_key

# PostgreSQL password
POSTGRES_PASSWORD=your_secure_password
```

### Optional Variables

```bash
# MCP server configuration
TRANSPORT=sse
HOST=0.0.0.0
PORT=8051

# LLM for summaries and contextual embeddings
MODEL_CHOICE=gpt-4o-mini

# RAG strategies (true/false)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Neo4j configuration (when USE_KNOWLEDGE_GRAPH=true)
NEO4J_PASSWORD=your_neo4j_password

# Admin tools
PGADMIN_PASSWORD=your_pgadmin_password
```

## Common Commands

### Starting Services

```bash
# Start basic stack
docker-compose up -d

# Start with knowledge graph
docker-compose --profile knowledge-graph up -d

# Start with admin tools
docker-compose --profile admin up -d

# Start everything
docker-compose --profile knowledge-graph --profile admin up -d

# Development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Managing Services

```bash
# View logs
docker-compose logs -f mcp-server
docker-compose logs -f postgres

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v

# Restart a specific service
docker-compose restart mcp-server

# Rebuild and restart
docker-compose up -d --build mcp-server
```

### Database Management

```bash
# Access PostgreSQL CLI
docker-compose exec postgres psql -U postgres -d crawl4ai_rag

# Access Neo4j CLI (when running)
docker-compose exec neo4j cypher-shell -u neo4j

# Backup PostgreSQL database
docker-compose exec postgres pg_dump -U postgres crawl4ai_rag > backup.sql

# Restore PostgreSQL database
docker-compose exec -T postgres psql -U postgres crawl4ai_rag < backup.sql
```

## Accessing Services

### MCP Server
- **URL**: `http://localhost:8051` (or configured port)
- **Health Check**: `http://localhost:8051/health` (if implemented)

### PostgreSQL Database
- **Host**: `localhost:5432`
- **Database**: `crawl4ai_rag`
- **Username**: `postgres`
- **Password**: From `POSTGRES_PASSWORD` env var

### pgAdmin (if enabled)
- **URL**: `http://localhost:5050`
- **Email**: `admin@crawl4ai.com`
- **Password**: From `PGADMIN_PASSWORD` env var

### Neo4j (if enabled)
- **Browser**: `http://localhost:7474`
- **Bolt**: `bolt://localhost:7687`
- **Username**: `neo4j`
- **Password**: From `NEO4J_PASSWORD` env var

## Troubleshooting

### Common Issues

#### 1. PostgreSQL Connection Failed
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Verify environment variables
docker-compose config
```

#### 2. MCP Server Can't Connect to Database
```bash
# Ensure PostgreSQL is healthy
docker-compose ps

# Check MCP server logs
docker-compose logs mcp-server

# Verify network connectivity
docker-compose exec mcp-server ping postgres
```

#### 3. pgvector Extension Missing
```bash
# Recreate PostgreSQL with proper image
docker-compose down
docker volume rm $(docker volume ls -q | grep postgres)
docker-compose up -d postgres
```

#### 4. Permission Issues (Development)
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
```

### Debugging

#### Enable Debug Logging
Add to your `.env` file:
```bash
# Enable debug logging in Python
PYTHONUNBUFFERED=1
```

#### Check Service Health
```bash
# Check all services status
docker-compose ps

# Check specific service health
docker-compose exec postgres pg_isready -U postgres
docker-compose exec neo4j cypher-shell -u neo4j "RETURN 1"
```

## Development Workflow

### 1. Code Changes
When developing with `docker-compose.dev.yml`, code changes are automatically reflected due to volume mounts.

### 2. Dependency Changes
If you modify `pyproject.toml`:
```bash
docker-compose build mcp-server
docker-compose up -d mcp-server
```

### 3. Database Schema Changes
If you modify `crawled_pages.sql`:
```bash
# Recreate database (WARNING: loses data)
docker-compose down
docker volume rm $(docker volume ls -q | grep postgres)
docker-compose up -d
```

### 4. Testing
```bash
# Run tests in container
docker-compose exec mcp-server python -m pytest tests/

# Or mount test volume in dev mode
```

## Production Considerations

### Security
- Change default passwords in production
- Use secrets management for sensitive values
- Configure firewalls appropriately
- Enable SSL/TLS for external access

### Performance
- Consider using connection pooling
- Monitor database performance
- Scale services horizontally if needed
- Use production-grade PostgreSQL configuration

### Backup
- Implement automated database backups
- Consider using external volumes for data persistence
- Test restore procedures regularly

### Monitoring
- Add health checks for all services
- Implement logging aggregation
- Monitor resource usage
- Set up alerting for failures
