from typing import Protocol, Dict
import json
import os

class StorageProvider(Protocol):
    def save_index(self, index_data: bytes) -> None:
        """Save FAISS index binary data"""
    
    def load_index(self) -> bytes:
        """Load FAISS index binary data"""
    
    def save_mapping(self, mapping: Dict[int, str]) -> None:
        """Save mapping dictionary"""
    
    def load_mapping(self) -> Dict[int, str]:
        """Load mapping dictionary"""
    
    def index_exists(self) -> bool:
        """Check if index exists"""
    
    def mapping_exists(self) -> bool:
        """Check if mapping exists"""
    
    def load_data(self, file_path: str) -> list:
        """Load JSON data from file"""

class FileSystemStorageProvider:
    def __init__(self, index_path: str, mapping_path: str):
        self.index_path: str = index_path
        self.mapping_path: str = mapping_path
    
    def save_index(self, index_data: bytes) -> None:
        with open(self.index_path, 'wb') as f:
            f.write(index_data)
    
    def load_index(self) -> bytes:
        with open(self.index_path, 'rb') as f:
            return f.read()
    
    def save_mapping(self, mapping: Dict[int, str]) -> None:
        with open(self.mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
    
    def load_mapping(self) -> Dict[int, str]:
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")
        with open(self.mapping_path, 'r') as f:
            loaded_mapping = json.load(f)
            return {int(k): v for k, v in loaded_mapping.items()}
    
    def index_exists(self) -> bool:
        return os.path.exists(self.index_path)
    
    def mapping_exists(self) -> bool:
        return os.path.exists(self.mapping_path)
    
    def load_data(self, file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
