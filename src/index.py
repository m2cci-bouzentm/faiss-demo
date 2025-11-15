import faiss
import numpy as np
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Step 1: Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
    return embeddings


# Step 2: Prepare data
names = [
    "John Smith",
    "Jane Doe", 
    "Bob Johnson",
    "Alice Williams",
    "Charlie Brown"
]

# Step 3: Generate embeddings
print("Generating embeddings using Google API...")
embeddings = generate_embeddings(names)
print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")

# Step 4: Convert to numpy array
embeddings_np = np.array(embeddings, dtype='float32')

# Step 5: Create FAISS index
dimension = len(embeddings[0])  
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)
print(f"Added {index.ntotal} vectors to FAISS index")

# Step 6: Save FAISS index
faiss.write_index(index, "names.faiss")
print("Saved FAISS index to names.faiss")


# Step 7: Save mapping separately
mapping = {}
for i, name in enumerate(names):
    mapping[i] = name

with open("mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)

print("Saved mapping to mapping.json")