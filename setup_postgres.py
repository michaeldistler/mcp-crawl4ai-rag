#!/usr/bin/env python3
"""
Setup script for migrating from Supabase to PostgreSQL.
This script helps set up the PostgreSQL database with the required schema.
"""
import os
import sys
import psycopg2
from pathlib import Path

def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        "POSTGRES_HOST",
        "POSTGRES_PORT", 
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_connection():
    """Test connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        conn.close()
        print("✅ Successfully connected to PostgreSQL")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        return False

def check_pgvector_extension():
    """Check if pgvector extension is available."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ pgvector extension is available")
        return True
    except Exception as e:
        print(f"❌ pgvector extension not available: {e}")
        print("   Install pgvector: https://github.com/pgvector/pgvector#installation")
        return False

def setup_database_schema():
    """Set up the database schema using the SQL file."""
    sql_file = Path(__file__).parent / "crawled_pages.sql"
    
    if not sql_file.exists():
        print(f"❌ SQL schema file not found: {sql_file}")
        return False
    
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        
        cursor = conn.cursor()
        
        # Read and execute the SQL file
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        cursor.execute(sql_content)
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Database schema created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create database schema: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up PostgreSQL for mcp-crawl4ai-rag\n")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment variables")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Test connection
    if not test_connection():
        sys.exit(1)
    
    # Check pgvector extension
    if not check_pgvector_extension():
        sys.exit(1)
    
    # Setup database schema
    if not setup_database_schema():
        sys.exit(1)
    
    print("\n🎉 PostgreSQL setup completed successfully!")
    print("\nNext steps:")
    print("1. Install the new dependencies: pip install -r requirements.txt")
    print("2. Update your .env file with PostgreSQL credentials")
    print("3. Test the MCP server with the new PostgreSQL backend")

if __name__ == "__main__":
    main()
