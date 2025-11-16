from typing import List, Tuple
import epitran
import Levenshtein


class PhoneticMatcher:
    def __init__(self, language_code: str = "fra-Latn"):
        self.epi = epitran.Epitran(language_code)
    
    def to_phonetic(self, text: str) -> str:
        return self.epi.transliterate(text)
    
    def calculate_distance(self, text1: str, text2: str) -> int:
        phonetic1 = self.to_phonetic(text1)
        phonetic2 = self.to_phonetic(text2)
        return Levenshtein.distance(phonetic1, phonetic2)
    
    def rank_by_phonetic_similarity(
        self, 
        query: str, 
        candidates: List[str]
    ) -> List[Tuple[str, int]]:
        query_phonetic = self.to_phonetic(query)
        
        results = []
        for candidate in candidates:
            candidate_phonetic = self.to_phonetic(candidate)
            distance = Levenshtein.distance(query_phonetic, candidate_phonetic)
            results.append((candidate, distance))
        
        results.sort(key=lambda x: x[1])
        return results

