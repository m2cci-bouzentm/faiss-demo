from typing import List, Dict, Tuple, Optional
import os
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from src.storage_provider import StorageProvider

load_dotenv()


class FaissManager:
    _model_name: str = "models/gemini-embedding-001"
    
    def __init__(self, storage_provider: StorageProvider):
        self.storage_provider: StorageProvider = storage_provider
        self.index: Optional[faiss.IndexFlatL2] = None
        self.mapping: Dict[int, str] = {}
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        
    def create_l2_index(self, dimension: int) -> None:
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_vectors(self, embeddings: List[List[float]], texts: List[str]) -> None:
        if self.index is None:
            raise ValueError("Index not created. Call create_l2_index() first.")
        
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        embeddings_np = np.array(embeddings, dtype='float32')
        
        start_idx = self.index.ntotal
        self.index.add(embeddings_np)
        
        for i, text in enumerate(texts):
            self.mapping[start_idx + i] = text
    
    def save(self) -> None:
        if self.index is None:
            raise ValueError("Index not created. Nothing to save.")
        
        index_data = faiss.serialize_index(self.index)
        self.storage_provider.save_index(index_data)
        self.storage_provider.save_mapping(self.mapping)
    
    def load(self) -> None:
        if not self.storage_provider.index_exists():
            raise FileNotFoundError("Index not found")
        
        if not self.storage_provider.mapping_exists():
            raise FileNotFoundError("Mapping not found")
        
        index_data = self.storage_provider.load_index()
        index_array = np.frombuffer(index_data, dtype=np.uint8)
        self.index = faiss.deserialize_index(index_array)
        self.mapping = self.storage_provider.load_mapping()
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("Index not created. Call create_l2_index() first.")
        
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add vectors first.")
        
        query_result = genai.embed_content(
            model=FaissManager._model_name,
            content=query,
            task_type="retrieval_query"
        )
        
        query_np = np.array([query_result['embedding']], dtype='float32')
        distances, indices = self.index.search(query_np, k)
        
        indices = indices[0]
        distances = distances[0]
        
        results: List[Tuple[str, float]] = []
        for i, faiss_id in enumerate(indices):
            if faiss_id in self.mapping:
                mapped_text = self.mapping[faiss_id]
                distance = float(distances[i])
                results.append((mapped_text, distance))
        
        return results
    
    def search_by_embedding(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("Index not created. Call create_l2_index() first.")
        
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add vectors first.")
        
        query_np = np.array([embedding], dtype='float32')
        distances, indices = self.index.search(query_np, k)
        
        indices = indices[0]
        distances = distances[0]
        
        results: List[Tuple[str, float]] = []
        for i, faiss_id in enumerate(indices):
            if faiss_id in self.mapping:
                mapped_text = self.mapping[faiss_id]
                distance = float(distances[i])
                results.append((mapped_text, distance))
        
        return results