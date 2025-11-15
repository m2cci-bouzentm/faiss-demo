import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
import json

index_path = "data/marques_index.faiss"
mapping_path = "data/marques_mapping.json"
output_path = "data/semantic_results.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)

faiss_manager.load()

test_queries = [
    "Selego",
    "Crédit",
]

all_results = []

for query in test_queries:
    results = faiss_manager.search(query, k=300)
    
    query_results = {
        "query": query,
        "results": [
            {"text": text, "distance": distance}
            for text, distance in results
        ]
    }
    all_results.append(query_results)
    
    print(f"Query: {query}")
    print(f"Saved {len(results)} semantic matches")
    print()

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Semantic results saved to {output_path}")

