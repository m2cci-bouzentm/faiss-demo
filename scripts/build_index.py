"""
Script for building FAISS indexes from raw data.

This script:
1. Loads raw data from JSON file
2. Extracts text fields (e.g., "Mark" field from trademark records)
3. Generates embeddings using Google Gemini API
4. Builds a FAISS L2 index for similarity search
5. Saves the index and mapping to disk
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.index_builder import IndexBuilder

json_file = "data/marques-francaises-latest-50k.json"
index_path = "data/marques_index.faiss"
mapping_path = "data/marques_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)
indexBuilder = IndexBuilder(faiss_manager)

data = storage.load_data(json_file)

marks = []
for record in data[:10000]:
    try:
        marks.append(record["Mark"])
    except KeyError:
        # Some records may not have "Mark" field 
        print(record["ApplicationNumber"], record["ApplicationDate"])
        pass

indexBuilder.build_index_from_texts(marks)