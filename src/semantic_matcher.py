import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticFaissEngine:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        
        logger.info(f"Loading Semantic Model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # 384 dimensions for MiniLM-L12-v2
        self.dimension = 384 
        self.index = faiss.IndexFlatIP(self.dimension)
        
        logger.info("Encoding texts for Semantic Search...")
        # encode returns numpy array by default
        embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        
        logger.info("Normalizing vectors...")
        faiss.normalize_L2(embeddings)
        
        logger.info("Indexing in FAISS...")
        self.index.add(embeddings)
        logger.info(f"Done! Indexed {self.index.ntotal} semantic vectors.")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Encode query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = query_vector.astype(np.float32)
        
        # Normalize
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = distances[0][i]
            if idx != -1 and idx < len(self.texts):
                results.append((self.texts[idx], float(score)))
                
        return results

