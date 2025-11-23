import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage_provider import FileSystemStorageProvider
from src.semantic_matcher import SemanticFaissEngine

# Configuration
JSON_FILE = "data/marques-francaises.json"
INDEX_PATH = "data/marques_index.faiss"
MAPPING_PATH = "data/marques_mapping.json"

storage = FileSystemStorageProvider(INDEX_PATH, MAPPING_PATH)

# Semantic queries focus on meaning rather than spelling
TEST_QUERIES = [
    "Green Energy",       # Should match energy companies, maybe "Vert", "Solaire", etc.
    "Fast Food",         # Should match restaurant chains
    "Computer Science",  # Should match tech/IT companies
    "Housing",           # Real estate, "Immobilier"
    "Bank",              # "Banque", "Credit", etc.
    "Organic Food",      # "Bio", "Nature"
]

def run_demo():
    print(f"--- 1. Loading Data from {JSON_FILE} ---")
    start_time = time.time()
    
    try:
        data = storage.load_data(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find file {JSON_FILE}")
        return

    all_brands = []
    missing_key_count = 0
    
    for record in data:
        if "Mark" in record and record["Mark"]:
            all_brands.append(record["Mark"])
        else:
            missing_key_count += 1
    
    # Deduplicate
    unique_brands = sorted(list(set(all_brands)))
    
    print(f"Loaded {len(data)} records.")
    print(f"Skipped {missing_key_count} records (missing 'Mark').")
    print(f"Final unique brands: {len(unique_brands)}")
    
    if len(unique_brands) == 0:
        print("CRITICAL: No brands loaded. Check your JSON key ('Mark').")
        return

    print(f"Data loading took {time.time() - start_time:.2f}s")
    print()

    print("--- 2. Initializing Semantic Engine (Sentence Transformers) ---")
    print("This will load a Deep Learning model (~100MB) and encode all brands.")
    start_time = time.time()
    
    # Initialize and index the engine
    engine = SemanticFaissEngine(unique_brands)
    
    print(f"Engine initialization complete in {time.time() - start_time:.2f}s")
    print("=" * 60)

    print("--- 3. Running Semantic Queries ---")
    for query in TEST_QUERIES:
        s_time = time.time()
        
        results = engine.search(query, k=5)
        
        dur = (time.time() - s_time) * 1000
        
        print(f"Query: '{query}'")
        print(f"Found matches in {dur:.2f}ms:")
        
        for i, (text, score) in enumerate(results, 1):
            # Semantic scores are usually lower than lexical ones (0.4-0.7 is often good)
            quality = "✅" if score > 0.6 else "⚠️" if score > 0.4 else "❌"
            print(f"   {i}. {quality} {text:<30} (Score: {score:.4f})")
        
        print("-" * 60)

if __name__ == "__main__":
    run_demo()

