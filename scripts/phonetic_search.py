import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.phonetic_matcher import PhoneticFaissEngine

index_path = "data/marques_index.faiss"
mapping_path = "data/marques_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)

test_queries = [
    "Selego",
    "Crédit",
]

def run_search():
    # Load brands from the existing mapping used by FaissManager
    # This ensures we are searching against the same set of brands as the semantic search
    print("Loading brands from mapping...")
    faiss_manager = FaissManager(storage_provider=storage)
    faiss_manager.load()
    
    all_brands = list(faiss_manager.mapping.values())
    # Filter out any Nones if they exist
    all_brands = [b for b in all_brands if b]
    
    print(f"Total brands loaded: {len(all_brands)}")
    
    print("Initializing Phonetic FAISS Engine...")
    engine = PhoneticFaissEngine(all_brands)
    print("Engine ready.")
    print()
    
    for query in test_queries:
        print(f"Query: {query}")
        
        results = engine.search(query, k=10)
        
        print(f"Phonetic vector matches (top 10):")
        for i, (text, score) in enumerate(results, 1):
            print(f"    {i}. {text} (Score: {score:.4f})")
        print()

if __name__ == "__main__":
    run_search()
