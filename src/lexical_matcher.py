import numpy as np
import logging
from tqdm import tqdm
import faiss
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LexicalFaissEngine:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.target_dimension = 128
        self.index = faiss.IndexFlatIP(self.target_dimension)
        
        self.legal_suffixes = {
            "sas", "sarl", "sa", "sci", "eurl", "snc", "ei", 
            "ltd", "inc", "corp", "cie", "groupe", "holding"
        }
        
        print("Step 1/4: Preprocessing texts...")
        self.preprocessed_texts = []
        for text in tqdm(texts, desc="Preprocessing"):
            self.preprocessed_texts.append(self._preprocess(text))
            
        print("Step 2/4: Learning Character N-Grams (TF-IDF)...")
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            min_df=2,
            dtype=np.float32
        )
        sparse_matrix = self.vectorizer.fit_transform(self.preprocessed_texts)
        
        print(f"Step 3/4: Compressing vectors (SVD) to {self.target_dimension} dims...")
        self.svd = TruncatedSVD(n_components=self.target_dimension, random_state=42)
        dense_vectors = self.svd.fit_transform(sparse_matrix)
        
        print("Step 4/4: Indexing in FAISS...")
        faiss.normalize_L2(dense_vectors)
        self.index.add(dense_vectors)
        print(f"Done! Indexed {self.index.ntotal} vectors.")

    def _preprocess(self, text: str) -> str:
        if not text:
            return ""
            
        text = text.lower()
        
        pattern = r'\b(' + '|'.join(re.escape(s) for s in self.legal_suffixes) + r')\b'
        text = re.sub(pattern, '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # 1. Preprocess query
        processed_query = self._preprocess(query)
        if not processed_query:
            return []
            
        # 2. Transform (TF-IDF)
        sparse_vec = self.vectorizer.transform([processed_query])
        
        # 3. Transform (SVD)
        dense_vec = self.svd.transform(sparse_vec)
        
        # 4. Normalize
        faiss.normalize_L2(dense_vec)
        
        # 5. FAISS Search
        distances, indices = self.index.search(dense_vec, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = distances[0][i]
            if idx != -1 and idx < len(self.texts):
                results.append((self.texts[idx], float(score)))
                
        return results

