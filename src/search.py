import faiss
import numpy as np
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in .env file")
genai.configure(api_key=GOOGLE_API_KEY)


test_queries = [
    "Bobby",              # Similar to "Bob Johnson"
    "Alicia",             # Similar to "Alice Williams"
    "Jonathan Smith",     # Similar to "John Smith"
    "Jane",               # Exact match with "Jane Doe"
]

index = faiss.read_index("names.faiss")
print(f"Loaded FAISS index with {index.ntotal} vectors")

with open("mapping.json", "r") as f:
    mapping = json.load(f)


query = test_queries[0]
print(f"\nQuery: '{query}'")

query_result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=query,
    task_type="retrieval_query"  
)
query_embedding = np.array([query_result['embedding']], dtype='float32')

# Search FAISS (get top 3 matches)
distances, indices = index.search(query_embedding, k=3)
print(f"Raw distances: {distances}")
print(f"Raw indices: {indices}")

indices = indices[0]
distances = distances[0]

for i, faiss_position in enumerate(indices):
    name = mapping[str(faiss_position)]
    distance = distances[i-1]
    print(f"    {i} {name} (FAISS position: {faiss_position}, distance: {distance})")