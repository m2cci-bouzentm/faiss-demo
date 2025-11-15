from typing import List
import os
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from faiss_manager import FaissManager
load_dotenv()


class IndexBuilder:
    _api_key: str = os.getenv("GOOGLE_API_KEY")
    _model_name: str = "models/gemini-embedding-001"
    
    def __init__(self, faiss_manager: FaissManager):
        self.faiss_manager: FaissManager = faiss_manager
        
        if not IndexBuilder._api_key:
            raise ValueError("Google API key required.")
        
        genai.configure(api_key=IndexBuilder._api_key)

    def build_index_from_texts(self, texts: List[str]) -> None:
        if len(texts) == 0:
            raise ValueError("Cannot build index with empty texts")
        
        embeddings = self._generate_embeddings(texts, batch_size=100)
        dimension = len(embeddings[0])
        
        self.faiss_manager.create_l2_index(dimension)
        self.faiss_manager.add_vectors(embeddings, texts)
        
        self.faiss_manager.save()
    
    def _generate_embeddings(self, texts: List[str], task_type: str = "retrieval_document", batch_size: int = 100) -> List[List[float]]:
        embeddings: List[List[float]] = []
        total_texts = len(texts)
        total_batches = (total_texts + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc="Generating embeddings", unit="batch") as pbar:
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                
                result = genai.embed_content(
                    model=IndexBuilder._model_name,
                    content=batch_texts,
                    task_type=task_type
                )
                
                # - BatchEmbeddingDict: embedding is list[list[float]]
                # - EmbeddingDict: embedding is list[float]
                if isinstance(result, dict) and 'embedding' in result:
                    embedding_data = result['embedding']
                    if isinstance(embedding_data, list):
                        if len(embedding_data) > 0 and isinstance(embedding_data[0], list):
                            embeddings.extend(embedding_data)
                        else:
                            embeddings.append(embedding_data)
                
                pbar.update(1)
                pbar.set_postfix({"embeddings": len(embeddings), "texts": min(i + batch_size, total_texts)})
        
        return embeddings