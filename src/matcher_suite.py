import logging
from collections import defaultdict
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

from src.phonetic_matcher import PhoneticFaissEngine
from src.lexical_matcher import LexicalFaissEngine
from src.semantic_matcher import SemanticFaissEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrandMatcherSuite:
    def __init__(self, brand_names: List[str]):
        self.brand_names = brand_names
        self.phonetic_engine = None
        self.lexical_engine = None
        self.semantic_engine = None
        
        self._init_engines()

    def _init_engines(self):
        """Initialize all three engines."""
        logger.info("Initializing BrandMatcherSuite Engines...")
        
        # Sequential initialization is safer to avoid memory/process conflicts with heavy models
        logger.info("--- Initializing Phonetic Engine ---")
        self.phonetic_engine = PhoneticFaissEngine(self.brand_names)
        
        logger.info("--- Initializing Lexical Engine ---")
        self.lexical_engine = LexicalFaissEngine(self.brand_names)
        
        logger.info("--- Initializing Semantic Engine ---")
        self.semantic_engine = SemanticFaissEngine(self.brand_names)
        
        logger.info("All engines initialized successfully.")

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Returns fused results using Reciprocal Rank Fusion (RRF).
        Output: List of (BrandName, RRF_Score, {breakdown_scores})
        """
        
        # Get results from all engines
        # We ask for k results from each engine, but maybe we should ask for more (e.g. 2*k) 
        # to increase overlap chances for RRF. Sticking to k as per basic prompt, 
        # but increasing it slightly internally is often better practice. 
        # User said "Get Top-K results from ... Engine", so I'll stick to k for now,
        # or maybe slightly higher to ensure good fusion. Let's use k*2 internally for better fusion.
        internal_k = k * 2
        
        phonetic_results, _ = self.phonetic_engine.search(query, k=internal_k)
        lexical_results = self.lexical_engine.search(query, k=internal_k)
        semantic_results = self.semantic_engine.search(query, k=internal_k)
        
        # RRF Constants
        k_const = 60
        
        # Fusion Logic
        rrf_scores = defaultdict(float)
        debug_info = defaultdict(dict)
        
        def process_results(results, engine_name):
            for rank, (name, score) in enumerate(results):
                # rank is 0-based here, so rank+1 is 1-based
                rrf_score = 1 / (k_const + (rank + 1))
                rrf_scores[name] += rrf_score
                
                # Store debug info
                if name not in debug_info:
                    debug_info[name] = {"RRF": 0.0, "Sources": []}
                
                debug_info[name]["Sources"].append(f"{engine_name} (rank={rank+1}, score={score:.4f})")
        
        process_results(phonetic_results, "Phonetic")
        process_results(lexical_results, "Lexical")
        process_results(semantic_results, "Semantic")
        
        # Format Output
        final_results = []
        for name, score in rrf_scores.items():
            final_results.append((name, score, debug_info[name]))
            
        # Sort by RRF score descending
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:k]

