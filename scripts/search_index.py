"""
Script for searching the FAISS index using semantic similarity.

This script:
1. Loads the pre-built FAISS index and mapping
2. Generates embeddings for query text using Google Gemini API
3. Performs similarity search to find the closest matches
4. Displays results with distances
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import google.generativeai as genai

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider

genai.configure(api_key=os.getenv("GOOGLE_API_KEY") )

index_path = "data/marques_index.faiss"
mapping_path = "data/marques_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)

faiss_manager.load()


test_queries = [
    "Selego",
    "Crédit",
]

for query in test_queries:
    query_result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = query_result['embedding']

    results = faiss_manager.search(query_embedding, k=3)

    print(f"Query: {query}")
    print(f"Found {len(results)} matches:")
    for text, distance in results:
        print(f"    {text} (distance: {distance})")
    print()