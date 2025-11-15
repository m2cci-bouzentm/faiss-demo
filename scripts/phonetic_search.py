import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_manager import FaissManager
from src.storage_provider import FileSystemStorageProvider
from src.phonetic_matcher import PhoneticMatcher

USE_SEMANTIC_SEARCH_RESULT = True

index_path = "data/marques_index.faiss"
mapping_path = "data/marques_mapping.json"
semantic_results_path = "data/semantic_results.json"

storage = FileSystemStorageProvider(index_path, mapping_path)

test_queries = [
    "Selego",
    "Crédit",
]

phonetic_matcher = PhoneticMatcher()

if USE_SEMANTIC_SEARCH_RESULT:
    semantic_results = storage.load_data(semantic_results_path)
    
    for query_data in semantic_results:
        query = query_data["query"]
        semantic_matches = query_data["results"]
        
        candidate_texts = [match["text"] for match in semantic_matches[:300]]
        
        phonetic_results = phonetic_matcher.rank_by_phonetic_similarity(query, candidate_texts)
        
        print(f"Query: {query}")
        print(f"Phonetic matches (top 10):")
        for text, distance in phonetic_results[:10]:
            phonetic_repr = phonetic_matcher.to_phonetic(text)
            print(f"    {text} → {phonetic_repr} (Levenshtein: {distance})")
        print()

else:
    faiss_manager = FaissManager(storage_provider=storage)
    faiss_manager.load()
    
    all_brands = list(faiss_manager.mapping.values())
    
    for query in test_queries:
        phonetic_results = phonetic_matcher.rank_by_phonetic_similarity(query, all_brands)
        
        print(f"Query: {query}")
        print(f"Phonetic matches (top 10):")
        for text, distance in phonetic_results[:10]:
            phonetic_repr = phonetic_matcher.to_phonetic(text)
            print(f"    {text} → {phonetic_repr} (Levenshtein: {distance})")
        print()

