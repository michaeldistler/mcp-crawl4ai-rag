"""
PostgreSQL utilities for the Crawl4AI MCP server.
Replaces Supabase functionality with direct PostgreSQL operations.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from urllib.parse import urlparse
import re
import time
from contextlib import contextmanager
import logging

from llm_providers import get_provider_manager

# Set up logging
logger = logging.getLogger(__name__)

class PostgresClient:
    """PostgreSQL client for vector operations and document storage."""
    
    def __init__(self):
        """Initialize PostgreSQL connection parameters from environment variables."""
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.database = os.getenv("POSTGRES_DB", "crawl4ai_rag")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD")
        
        if not self.password:
            raise ValueError("POSTGRES_PASSWORD must be set in environment variables")
    
    @contextmanager
    def get_connection(self):
        """
        Get a PostgreSQL connection with proper error handling.
        
        Yields:
            psycopg2 connection object
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                cursor_factory=RealDictCursor
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Optional[List[Dict]]:
        """
        Execute a query with proper error handling.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, None otherwise
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return [dict(row) for row in cursor.fetchall()]
                conn.commit()
                return None


def get_postgres_client() -> PostgresClient:
    """
    Get a PostgreSQL client instance.
    
    Returns:
        PostgreSQL client instance
    """
    return PostgresClient()


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts using the configured provider.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    try:
        provider_manager = get_provider_manager()
        return provider_manager.create_embeddings_batch(texts)
    except Exception as e:
        logger.error(f"Error creating batch embeddings: {e}")
        # Return zero embeddings as fallback
        provider_manager = get_provider_manager()
        embedding_dim = provider_manager.get_embedding_dimension()
        return [[0.0] * embedding_dim] * len(texts)


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using the configured provider.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        provider_manager = get_provider_manager()
        return provider_manager.create_embedding(text)
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        # Return zero embedding as fallback
        provider_manager = get_provider_manager()
        embedding_dim = provider_manager.get_embedding_dimension()
        return [0.0] * embedding_dim


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Use the provider manager to generate contextual information
        provider_manager = get_provider_manager()
        context = provider_manager.create_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            model=model_choice,
            temperature=0.3,
            max_tokens=200
        )
        
        # Combine the context with the original chunk
        contextual_text = f"{context.strip()}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        logger.error(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False


def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)


def add_documents_to_postgres(
    client: PostgresClient, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the PostgreSQL crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        client: PostgreSQL client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs
    try:
        if unique_urls:
            placeholders = ','.join(['%s'] * len(unique_urls))
            delete_query = f"DELETE FROM crawled_pages WHERE url IN ({placeholders})"
            client.execute_query(delete_query, tuple(unique_urls))
            logger.info(f"Deleted existing records for {len(unique_urls)} URLs")
    except Exception as e:
        logger.error(f"Error deleting existing records: {e}")
        # Continue with insertion even if deletion fails
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    logger.info(f"Use contextual embeddings: {use_contextual_embeddings}")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])
            
            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                logger.warning(f"Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)
        
        # Prepare batch data for insertion
        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract source_id from URL
            parsed_url = urlparse(batch_urls[j])
            source_id = parsed_url.netloc or parsed_url.path
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, batch_embeddings[j])) + ']'
            
            # Prepare data for insertion
            batch_data.append((
                batch_urls[j],
                batch_chunk_numbers[j],
                contextual_contents[j],
                json.dumps({
                    "chunk_size": len(contextual_contents[j]),
                    **batch_metadatas[j]
                }),
                source_id,
                embedding_str
            ))
        
        # Insert batch into PostgreSQL with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        # Get embedding dimension from provider manager
        provider_manager = get_provider_manager()
        embedding_dim = provider_manager.get_embedding_dimension()
        
        insert_query = f"""
            INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector({embedding_dim}))
        """
        
        for retry in range(max_retries):
            try:
                with client.get_connection() as conn:
                    with conn.cursor() as cursor:
                        execute_batch(cursor, insert_query, batch_data)
                    conn.commit()
                logger.info(f"Successfully inserted batch {i//batch_size + 1}")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Error inserting batch (attempt {retry + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Try inserting records one by one as a last resort
                    logger.info("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.execute_query(insert_query, record)
                            successful_inserts += 1
                        except Exception as individual_error:
                            logger.error(f"Failed to insert individual record for URL {record[0]}: {individual_error}")
                    
                    if successful_inserts > 0:
                        logger.info(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")


def search_documents(
    client: PostgresClient, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in PostgreSQL using vector similarity.
    
    Args:
        client: PostgreSQL client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_filter: Optional source ID filter
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    try:
        # Convert embedding list to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Get embedding dimension from provider manager
        provider_manager = get_provider_manager()
        embedding_dim = provider_manager.get_embedding_dimension()
        
        # Add metadata filter
        if filter_metadata:
            filter_json = json.dumps(filter_metadata)
        else:
            filter_json = '{}'
        
        # Build the query with proper vector casting - use dynamic dimension
        search_query = f"""
            SELECT * FROM match_crawled_pages(%s::vector({embedding_dim}), %s, %s::jsonb, %s)
        """
        
        # Prepare query parameters
        params = [embedding_str, match_count, filter_json, source_filter]
        
        # Execute the search
        result = client.execute_query(search_query, tuple(params), fetch=True)
        return result or []
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []


def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        logger.info("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks


def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        provider_manager = get_provider_manager()
        response_text = provider_manager.create_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            model=model_choice,
            temperature=0.3,
            max_tokens=100
        )
        
        return response_text.strip()
    
    except Exception as e:
        logger.error(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def add_code_examples_to_postgres(
    client: PostgresClient,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """
    Add code examples to the PostgreSQL code_examples table in batches.
    
    Args:
        client: PostgreSQL client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return
        
    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    try:
        if unique_urls:
            placeholders = ','.join(['%s'] * len(unique_urls))
            delete_query = f"DELETE FROM code_examples WHERE url IN ({placeholders})"
            client.execute_query(delete_query, tuple(unique_urls))
            logger.info(f"Deleted existing code examples for {len(unique_urls)} URLs")
    except Exception as e:
        logger.error(f"Error deleting existing code examples: {e}")
    
    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []
        
        # Create combined texts for embedding (code + summary)
        for j in range(i, batch_end):
            combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
            batch_texts.append(combined_text)
        
        # Create embeddings for the batch
        embeddings = create_embeddings_batch(batch_texts)
        
        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        for embedding in embeddings:
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
            else:
                logger.warning("Zero or invalid embedding detected, creating new one...")
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(batch_texts[len(valid_embeddings)])
                valid_embeddings.append(single_embedding)
        
        # Prepare batch data
        batch_data = []
        for j, embedding in enumerate(valid_embeddings):
            idx = i + j
            
            # Extract source_id from URL
            parsed_url = urlparse(urls[idx])
            source_id = parsed_url.netloc or parsed_url.path
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            batch_data.append((
                urls[idx],
                chunk_numbers[idx],
                code_examples[idx],
                summaries[idx],
                json.dumps(metadatas[idx]),
                source_id,
                embedding_str
            ))
        
        # Insert batch into PostgreSQL with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        insert_query = """
            INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
        """
        
        for retry in range(max_retries):
            try:
                with client.get_connection() as conn:
                    with conn.cursor() as cursor:
                        execute_batch(cursor, insert_query, batch_data)
                    conn.commit()
                logger.info(f"Successfully inserted code examples batch {i//batch_size + 1}")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Error inserting code examples batch (attempt {retry + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to insert code examples batch after {max_retries} attempts: {e}")
                    # Try inserting records one by one as a last resort
                    logger.info("Attempting to insert code examples individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.execute_query(insert_query, record)
                            successful_inserts += 1
                        except Exception as individual_error:
                            logger.error(f"Failed to insert individual code example for URL {record[0]}: {individual_error}")
                    
                    if successful_inserts > 0:
                        logger.info(f"Successfully inserted {successful_inserts}/{len(batch_data)} code examples individually")
        
        logger.info(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")


def update_source_info(client: PostgresClient, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources table.
    
    Args:
        client: PostgreSQL client
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        # Try to update existing source
        update_query = """
            UPDATE sources 
            SET summary = %s, total_word_count = %s, updated_at = NOW()
            WHERE source_id = %s
        """
        
        with client.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, (summary, word_count, source_id))
                rows_updated = cursor.rowcount
                
                # If no rows were updated, insert new source
                if rows_updated == 0:
                    insert_query = """
                        INSERT INTO sources (source_id, summary, total_word_count)
                        VALUES (%s, %s, %s)
                    """
                    cursor.execute(insert_query, (source_id, summary, word_count))
                    logger.info(f"Created new source: {source_id}")
                else:
                    logger.info(f"Updated source: {source_id}")
                
                conn.commit()
            
    except Exception as e:
        logger.error(f"Error updating source {source_id}: {e}")


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    
    This function uses the OpenAI API to generate a concise summary of the source content.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        # Use the provider manager to generate the summary
        provider_manager = get_provider_manager()
        summary = provider_manager.create_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            model=model_choice,
            temperature=0.3,
            max_tokens=150
        )
        
        # Extract the generated summary and strip whitespace
        summary = summary.strip()
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        logger.error(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary


def search_code_examples(
    client: PostgresClient, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in PostgreSQL using vector similarity.
    
    Args:
        client: PostgreSQL client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    # Since code examples are embedded with their summaries, we should make the query more descriptive
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)
    
    try:
        # Convert embedding list to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Prepare query parameters
        params = [embedding_str, match_count]
        
        # Build the query with proper vector casting
        search_query = """
            SELECT * FROM match_code_examples(%s::vector, %s, %s, %s)
        """
        
        # Add metadata filter
        if filter_metadata:
            filter_json = json.dumps(filter_metadata)
        else:
            filter_json = '{}'
        params.append(filter_json)
        
        # Add source filter
        params.append(source_id)
        
        # Execute the search
        result = client.execute_query(search_query, tuple(params), fetch=True)
        return result or []
        
    except Exception as e:
        logger.error(f"Error searching code examples: {e}")
        return []
