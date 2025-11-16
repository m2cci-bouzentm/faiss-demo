"""
Script for building FAISS indexes from images.

This script:
1. Loads image URLs from JSON file
2. Downloads and processes images
3. Generates embeddings using CLIP
4. Builds a FAISS L2 index for similarity search
5. Saves the index and mapping to disk
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.image_embedder import ImageEmbedder

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

json_file = "data/image_urls.json"
index_path = "data/images_index.faiss"
mapping_path = "data/images_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)
image_embedder = ImageEmbedder()

with open(json_file, 'r') as f:
    image_urls = json.load(f)

dimension = image_embedder.get_embedding_dimension()
faiss_manager.create_l2_index(dimension)

embeddings = []
valid_urls = []

for url in tqdm(image_urls, desc="Processing images"):
    try:
        image = load_image_from_url(url)
        embedding = image_embedder.embed_image(image)
        embeddings.append(embedding.tolist())
        valid_urls.append(url)
    except Exception as e:
        pass

faiss_manager.add_vectors(embeddings, valid_urls)
faiss_manager.save()