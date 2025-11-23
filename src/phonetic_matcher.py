from typing import List, Tuple, Dict
import numpy as np
import os
import logging
from tqdm import tqdm
import faiss
import sys

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from lingua import Language, LanguageDetectorBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('phonemizer').setLevel(logging.ERROR)
LANGUAGE_TO_ESPEAK: Dict[Language, str] = {
    Language.FRENCH: 'fr-fr',
    Language.ENGLISH: 'en-us',
    Language.SPANISH: 'es',
    Language.GERMAN: 'de',
    Language.ITALIAN: 'it',
    Language.PORTUGUESE: 'pt',
}

language_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.FRENCH, Language.SPANISH, 
    Language.GERMAN, Language.ITALIAN, Language.PORTUGUESE
).with_minimum_relative_distance(0.15).build()

class PhoneticFaissEngine:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.target_dimension = 128 
        self.index = faiss.IndexFlatIP(self.target_dimension)
        
        lib_path = os.environ.get('PHONEMIZER_ESPEAK_LIBRARY', None)
        
        if not lib_path:
            possible_paths = [
                '/opt/homebrew/lib/libespeak-ng.dylib',
                '/usr/local/lib/libespeak-ng.dylib',
                '/opt/homebrew/lib/libespeak.dylib',
                '/usr/local/lib/libespeak.dylib'
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    lib_path = p
                    logger.info(f"Found espeak library at: {lib_path}")
                    break
        
        if lib_path:
            try:
                EspeakWrapper.set_library(lib_path)
            except Exception as e:
                logger.error(f"Failed to set library: {e}")
        else:
            logger.warning("Could not find espeak library automatically. Phonemization might fail.")

        self.backends = {}
        unique_codes = set(LANGUAGE_TO_ESPEAK.values())
        unique_codes.add('fr-fr')
        
        print("Initializing Espeak backends...")
        for code in unique_codes:
            try:
                self.backends[code] = EspeakBackend(
                    language=code,
                    preserve_punctuation=False,
                    with_stress=False
                )
            except Exception as e:
                # Log the error so we know if initialization fails
                logger.warning(f"Backend init failed for {code}: {e}")

        # --- 2. GENERATE PHONETICS ---
        print("Step 1/4: Detecting Languages & Phonemizing...")
        self.ipa_corpus = []
        
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc="Phonemizing"):
            batch = texts[i:i+batch_size]
            batch_ipas = self._batch_to_ipa(batch)
            self.ipa_corpus.extend(batch_ipas)

        # --- DEBUG CHECK ---
        if not any(self.ipa_corpus) or len(self.ipa_corpus) == 0:
            logger.error("CRITICAL: IPA Corpus is empty. Espeak failed to phonemize everything.")
            # Print the first few items to prove they are empty
            print(f"First 5 IPA results: {self.ipa_corpus[:5]}")
            raise RuntimeError("Espeak configuration error - see logs above.")

        print("Step 2/4: Learning Phonetic N-Grams (TF-IDF)...")
        self.vectorizer = TfidfVectorizer(
            analyzer='char', 
            ngram_range=(2, 4), 
            min_df=2,           
            dtype=np.float32
        )
        sparse_matrix = self.vectorizer.fit_transform(self.ipa_corpus)

        print(f"Step 3/4: Compressing vectors (SVD) to {self.target_dimension} dims...")
        self.svd = TruncatedSVD(n_components=self.target_dimension, random_state=42)
        dense_vectors = self.svd.fit_transform(sparse_matrix)

        faiss.normalize_L2(dense_vectors)

        print("Step 4/4: Indexing in FAISS...")
        self.index.add(dense_vectors)
        print(f"Done! Indexed {self.index.ntotal} vectors.")

    def _batch_to_ipa(self, texts: List[str]) -> List[str]:
        lang_groups = {}
        fallback = 'fr-fr'
        
        for idx, text in enumerate(texts):
            try:
                det = language_detector.detect_language_of(text)
                code = LANGUAGE_TO_ESPEAK.get(det, fallback)
            except:
                code = fallback
            if code not in lang_groups: lang_groups[code] = []
            lang_groups[code].append((idx, text))

        results = [""] * len(texts)

        for code, items in lang_groups.items():
            indices = [x[0] for x in items]
            batch_txt = [x[1] for x in items]
            
            backend = self.backends.get(code, self.backends.get('fr-fr'))
            if not backend: 
                print(f"No backend for {code}")
                continue
            
            try:
                ipas = backend.phonemize(batch_txt, strip=True, njobs=1)
                for i, ipa in enumerate(ipas):
                    results[indices[i]] = ipa
            except Exception as e:
                print(f"\n[ERROR] Phonemize failed for {code}: {e}")
                sys.exit(1) 
                
        return results

    def search(self, query: str, k: int = 5) -> Tuple[List[Tuple[str, float]], str]:
        """
        Returns:
            - List of (BrandName, Score)
            - The Detected Language Code (e.g., 'en-us', 'fr-fr')
        """
        try:
            detected = language_detector.detect_language_of(query)
            lang_code = LANGUAGE_TO_ESPEAK.get(detected, 'fr-fr')
        except:
            lang_code = 'fr-fr'

        backend = self.backends.get(lang_code, self.backends.get('fr-fr'))
        
        if not backend: 
            return [], lang_code

        try:
            ipa_list = backend.phonemize([query], strip=True, njobs=1)
            query_ipa = ipa_list[0] if ipa_list else ""
        except Exception as e:
            logger.error(f"Query phonemization failed: {e}")
            return [], lang_code
        
        if not query_ipa: 
            return [], lang_code

        # 3. Vectorize
        sparse_vec = self.vectorizer.transform([query_ipa])
        dense_vec = self.svd.transform(sparse_vec)
        faiss.normalize_L2(dense_vec)
        
        # 4. Search
        distances, indices = self.index.search(dense_vec, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = distances[0][i]
            if idx != -1 and idx < len(self.texts):
                results.append((self.texts[idx], float(score)))
                
        return results, lang_code