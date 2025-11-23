import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import json
import requests
import tempfile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.audio_embedder import AudioEmbedder

def download_audio(url: str, temp_path: str) -> bool:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        return False

json_file = "data/audio_urls.json"
index_path = "data/audio_index.faiss"
mapping_path = "data/audio_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)
audio_embedder = AudioEmbedder()

with open(json_file, 'r') as f:
    audio_urls = json.load(f)

dimension = audio_embedder.get_embedding_dimension()
print(f"Expected embedding dimension from model config: {dimension}")

embeddings = []
valid_urls = []
failed_urls = []

for url in tqdm(audio_urls, desc="Processing audio files"):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        if download_audio(url, temp_path):
            print(f"\nProcessing: {url}")
            embedding = audio_embedder.embed_audio(temp_path)
            print(f"Embedding shape: {embedding.shape}")
            embeddings.append(embedding.tolist())
            valid_urls.append(url)
            print(f"✓ Success")
        else:
            print(f"\n✗ Failed to download: {url}")
            failed_urls.append(url)
    except Exception as e:
        print(f"\n✗ Error processing {url}: {e}")
        failed_urls.append(url)
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Total URLs attempted: {len(audio_urls)}")
print(f"Successfully processed: {len(valid_urls)}")
print(f"Failed: {len(failed_urls)}")
print(f"{'='*60}\n")

if failed_urls:
    print("Failed URLs:")
    for url in failed_urls:
        print(f"  - {url}")
    print()

if embeddings:
    actual_dimension = len(embeddings[0])
    print(f"Actual embedding dimension: {actual_dimension}")
    print(f"Number of embeddings: {len(embeddings)}")
    
    if actual_dimension != dimension:
        print(f"WARNING: Dimension mismatch! Using actual dimension: {actual_dimension}")
        dimension = actual_dimension
    
    faiss_manager.create_l2_index(dimension)
    faiss_manager.add_vectors(embeddings, valid_urls)
    faiss_manager.save()
    
    print(f"\n✓ Audio index built successfully!")
    print(f"✓ Total audio files indexed: {len(valid_urls)}")
else:
    print("✗ No embeddings generated! Check the errors above.")
