"""
Simple script to embed a single image using CLIP and log the vectors.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_embedder import ImageEmbedder

image_path = "http://images.cocodataset.org/val2017/000000039769.jpg"

print(f"Loading image from: {image_path}")

image = Image.open(requests.get(image_path, stream=True).raw)

print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")

clip_embedder = ImageEmbedder()
print(f"\nEmbedding dimension: {clip_embedder.get_embedding_dimension()}")

print("\nGenerating embedding...")
embedding = clip_embedder.embed_image(image)

print(f"\nEmbedding shape: {embedding.shape}")
print(f"Embedding dtype: {embedding.dtype}")
print(f"\nEmbedding vector:")
print(embedding)


