"""
Script for searching the FAISS image index using similarity.

This script:
1. Loads the pre-built FAISS image index and mapping
2. Generates embeddings for query image using CLIP
3. Performs similarity search to find the closest matches
4. Displays results with distances
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.image_embedder import ImageEmbedder

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

index_path = "data/images_index.faiss"
mapping_path = "data/images_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)
image_embedder = ImageEmbedder()

faiss_manager.load()

test_queries = [
    "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800&h=600&fit=crop",
]

for query_url in test_queries:
    query_image = load_image_from_url(query_url)
    query_embedding = image_embedder.embed_image(query_image)
    results = faiss_manager.search_by_embedding(query_embedding, k=5)
    
    print(f"Query: {query_url}")
    print(f"Found {len(results)} matches:")
    for image_url, distance in results:
        print(f"    {image_url} (distance: {distance})")
    print()

