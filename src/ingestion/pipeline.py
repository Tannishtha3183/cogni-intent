import os
import sys

import pandas as pd

# Add the root directory to Python's path so we can import our modules correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.encoder import TextEncoder
from src.storage.vector_client import MemoryStore


def run_ingestion():
    print("--- Starting Data Ingestion Pipeline ---")
    encoder = TextEncoder()
    db = MemoryStore()

    print("Reading synthetic patient logs...")
    df = pd.read_csv("data/synthetic_logs.csv")

    for index, row in df.iterrows():
        doc_id = row['id']
        text = row['text']

        # 1. Convert text to vector
        vector = encoder.encode(text)

        # 2. Build the strict MLOps metadata schema
        metadata = {
            "patient_id": row['patient_id'],
            "timestamp": float(row['timestamp']),
            "relevance_score": 1.0,
            "is_archived": False,       # Crucial for our Soft-Eviction policy
            "is_consolidated": False
        }

        # 3. Store in the database
        db.insert_memory(doc_id, vector, text, metadata)
        print(f"Ingested -> {text[:60]}...")

    print("\n--- Ingestion Complete! All memories stored in ChromaDB. ---")

if __name__ == "__main__":
    run_ingestion()
