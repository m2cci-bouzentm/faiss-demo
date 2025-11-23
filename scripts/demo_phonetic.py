import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage_provider import FileSystemStorageProvider
from src.phonetic_matcher import PhoneticFaissEngine

# Configuration
JSON_FILE = "data/marques-francaises.json"
INDEX_PATH = "data/marques_index.faiss"
MAPPING_PATH = "data/marques_mapping.json"

storage = FileSystemStorageProvider(INDEX_PATH, MAPPING_PATH)

TEST_QUERIES = [
    "Collectif",
    "Beeldi",
    "Subway",
    "Tech Solutions", 
    "Pharmacie",
    "Axa"
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
    
    unique_brands = sorted(list(set(all_brands)))
    
    print(f"Loaded {len(data)} records.")
    print(f"Skipped {missing_key_count} records (missing 'Mark').")
    print(f"Final unique brands: {len(unique_brands)}")
    
    if len(unique_brands) == 0:
        print("CRITICAL: No brands loaded. Check your JSON key ('Mark').")
        return

    print(f"Data loading took {time.time() - start_time:.2f}s")
    print()

    print("--- 2. Initializing Phonetic Engine (Phonemization + TF-IDF + SVD) ---")
    print("This may take a minute for 50k brands...")
    start_time = time.time()
    
    engine = PhoneticFaissEngine(unique_brands)
    
    print(f"Engine training complete in {time.time() - start_time:.2f}s")
    print("=" * 60)


    print("--- 3. Running Queries ---")
    for query in TEST_QUERIES:
        s_time = time.time()
        
        results, detected_lang = engine.search(query, k=10)
        
        dur = (time.time() - s_time) * 1000
        
        print(f"Query: '{query}' (Detected: {detected_lang})")
        print(f"Found matches in {dur:.2f}ms:")
        
        for i, (text, score) in enumerate(results, 1):
            quality = "✅" if score > 0.85 else "⚠️" if score > 0.8 else "❌"
            print(f"   {i}. {quality} {text:<30} (Score: {score:.4f})")
        
        print("-" * 60)

if __name__ == "__main__":
    run_demo()