import chromadb
import os
from typing import Dict, Any, List, Optional
from src.core.logger import get_logger

logger = get_logger(__name__)

class MemoryStore:
    """
    Diachronic Vector Memory Store backed by persistent ChromaDB.
    Implements soft-eviction policies to maintain complete historical lineage
    for long-term AI agents without clinical data loss.
    """
    def __init__(self, db_path: str = "data/chroma_db"):
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="patient_memories",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to ChromaDB at '{db_path}' (Collection: patient_memories)")

    def insert_memory(self, doc_id: str, vector: List[float], text: str, metadata: Dict[str, Any]):
        """Appends a new vector document with MLOps diachronic metadata."""
        self.collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata]
        )

    def fetch_active_memories(self) -> Dict[str, Any]:
        """Retrieves only active, un-archived memories currently in the agent's context pool."""
        return self.collection.get(
            where={"is_archived": False},
            include=['embeddings', 'documents', 'metadatas']
        )

    def fetch_all(self) -> Dict[str, Any]:
        """Retrieves all memories including historical soft-evicted records for MLOps audits."""
        return self.collection.get(
            include=['embeddings', 'documents', 'metadatas']
        )

    def soft_delete(self, doc_id: str):
        """
        Soft-evicts a memory by setting is_archived=True.
        The vector remains stored for historical audit trails and legal compliance.
        """
        existing = self.collection.get(ids=[doc_id])
        if existing and existing['metadatas']:
            meta = existing['metadatas'][0]
            meta['is_archived'] = True
            self.collection.update(ids=[doc_id], metadatas=[meta])
            logger.info(f"Memory {doc_id} soft-evicted (is_archived=True).")

    def query_active(self, query_vector: List[float], n_results: int = 3, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Simulates an AI agent's semantic search retrieval against the active memory pool.
        Demonstrates the contrast before and after healing.
        """
        where_clause = {"is_archived": False}
        if patient_id:
            where_clause = {"$and": [{"is_archived": False}, {"patient_id": patient_id}]}

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(n_results, max(1, self.collection.count())),
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )

            hits = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    hits.append({
                        "id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else 0.0
                    })
            return hits
        except Exception as e:
            logger.error(f"Error querying active memories: {e}")
            return []

    def get_lineage_dag(self, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts the Directed Acyclic Graph (DAG) of memory evolution.
        Maps parent obsolete memories to superseding Golden Vectors.
        """
        all_mems = self.fetch_all()
        nodes = []
        edges = []

        if not all_mems or not all_mems['ids']:
            return {"nodes": [], "edges": []}

        for i, doc_id in enumerate(all_mems['ids']):
            meta = all_mems['metadatas'][i]
            if patient_id and meta.get('patient_id') != patient_id:
                continue

            is_golden = meta.get('is_consolidated', False)
            is_archived = meta.get('is_archived', False)

            status = "golden" if is_golden else ("archived" if is_archived else "active")

            nodes.append({
                "id": doc_id,
                "label": all_mems['documents'][i][:45] + ("..." if len(all_mems['documents'][i]) > 45 else ""),
                "full_text": all_mems['documents'][i],
                "patient_id": meta.get('patient_id'),
                "status": status,
                "timestamp": meta.get('timestamp')
            })

            # Check for supersedes edges
            supersedes_str = meta.get('supersedes_ids', "")
            if supersedes_str:
                for old_id in supersedes_str.split(","):
                    old_id = old_id.strip()
                    if old_id:
                        edges.append({
                            "from": old_id,
                            "to": doc_id,
                            "relation": "superseded_by"
                        })

        return {"nodes": nodes, "edges": edges}