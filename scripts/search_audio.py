import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import requests
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.audio_embedder import AudioEmbedder

index_path = "data/audio_index.faiss"
mapping_path = "data/audio_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)
audio_embedder = AudioEmbedder()

faiss_manager.load()

query_urls = [
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
]

for query_num, test_audio_url in enumerate(query_urls, 1):
    print(f"{'='*60}")
    print(f"Query #{query_num}: {test_audio_url}")
    print(f"{'='*60}")
    
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        print("Downloading query audio...")
        response = requests.get(test_audio_url, timeout=30)
        response.raise_for_status()
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        print("Generating embedding for query audio...")
        query_embedding = audio_embedder.embed_audio(temp_path)
        
        print("Searching for similar audio...\n")
        results = faiss_manager.search_by_embedding(query_embedding, k=5)
        
        print(f"Top {len(results)} most similar audio files:")
        for i, (audio_url, distance) in enumerate(results, 1):
            print(f"{i}. {audio_url}")
            print(f"   Distance: {distance:.4f}\n")
        
    except Exception as e:
        print(f"Error processing query: {e}\n")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print()
