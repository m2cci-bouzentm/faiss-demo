import json
import sys
import os
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.matcher_suite import BrandMatcherSuite

def load_brands(filepath):
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    brands = [item.get("Mark") for item in data if item.get("Mark")]
    brands = list(set(brands))
    print(f"Loaded {len(brands)} unique brands.")
    return brands

def main():
    data_path = project_root / "data" / "marques-francaises.json"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    # 1. Load Data
    brands = load_brands(data_path)
    
    # Use a subset for quicker demo if needed, but user said "Load the 50k JSON"
    # We will use all of them.
    
    # 2. Initialize Suite
    suite = BrandMatcherSuite(brands)
    
    # 3. Run Test Queries
    test_queries = [
        "Homeland",      # Expected: "Home Land" (Lexical/Phonetic)
        "Green Energy",  # Expected: "Vert Energie" (Semantic)
        "Gooogle",       # Expected: "Google" (Lexical)
        "Beeldi"         # Expected: "Buildy" (Phonetic)
    ]
    
    print("\n" + "="*50)
    print("STARTING DEMO SUITE SEARCH")
    print("="*50)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = suite.search(query, k=5)
        
        for rank, (name, score, debug) in enumerate(results, 1):
            print(f"  {rank}. {name} (RRF: {score:.4f})")
            # Print sources for debugging visibility
            sources = ", ".join(debug["Sources"])
            print(f"     Found by: {sources}")

if __name__ == "__main__":
    main()

