import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.phonetic_matcher import PhoneticMatcher

USE_SEMANTIC_SEARCH_RESULT = False

index_path = "data/marques_index.faiss"
mapping_path = "data/marques_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)
faiss_manager.load()

phonetic_matcher = PhoneticMatcher()

test_queries = [
    "Selego",
    "Crédit",
    "Larodj",
]

if USE_SEMANTIC_SEARCH_RESULT:
    print("Mode: Using semantic search results (top 300)")
    print("=" * 60)
    print()
    
    for query in test_queries:
        print(f"Query: {query}")
        query_phonetic = phonetic_matcher.to_phonetic(query)
        print(f"Query phonetic: {query_phonetic}")
        print()
        
        semantic_results = faiss_manager.search(query, k=300)
        candidate_texts = [text for text, _ in semantic_results]
        
        phonetic_results = phonetic_matcher.rank_by_phonetic_similarity(query, candidate_texts)
        
        print(f"Top 10 phonetic matches (from 300 semantic candidates):")
        for i, (text, distance) in enumerate(phonetic_results[:10], 1):
            text_phonetic = phonetic_matcher.to_phonetic(text)
            print(f"  {i}. {text}")
            print(f"     → {text_phonetic} (Levenshtein: {distance})")
        print()
        print("-" * 60)
        print()

else:
    print("Mode: Full scan on all brands")
    print("=" * 60)
    print()
    
    all_brands = list(faiss_manager.mapping.values())
    print(f"Total brands in database: {len(all_brands)}")
    print()
    
    for query in test_queries:
        print(f"Query: {query}")
        query_phonetic = phonetic_matcher.to_phonetic(query)
        print(f"Query phonetic: {query_phonetic}")
        print()
        
        phonetic_results = phonetic_matcher.rank_by_phonetic_similarity(query, all_brands)
        
        print(f"Top 10 phonetic matches (full scan):")
        for i, (text, distance) in enumerate(phonetic_results[:10], 1):
            text_phonetic = phonetic_matcher.to_phonetic(text)
            print(f"  {i}. {text}")
            print(f"     → {text_phonetic} (Levenshtein: {distance})")
        print()
        print("-" * 60)
        print()

