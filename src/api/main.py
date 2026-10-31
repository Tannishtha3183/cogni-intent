from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import sys
import uuid
import time

# Add root directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.storage.vector_client import MemoryStore
from src.mlops.drift_detector import DriftDetector
from src.mlops.healer import MemoryHealer
from src.ingestion.encoder import TextEncoder
from src.ingestion.pipeline import run_ingestion
from src.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Cogni-Intent MLOps Engine",
    description="Autonomous MLOps Platform & Cognitive Immune System for Long-Term AI Agent Memory",
    version="2.0.0"
)

# Enable CORS for all origins to allow browser client interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core engine singletons
db = MemoryStore()
detector = DriftDetector()
healer = MemoryHealer()
encoder = TextEncoder()

# Pydantic request / response models
class QueryRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None
    top_k: int = 3

class CustomInjectRequest(BaseModel):
    patient_id: str
    scenario_name: str
    records: List[str]

@app.get("/api/health")
def health_check():
    """Returns real-time engine telemetry and vector store connectivity."""
    all_mems = db.fetch_all()
    active_count = len(db.fetch_active_memories().get('ids', []))
    total_count = len(all_mems.get('ids', []))
    return {
        "status": "healthy",
        "service": "Cogni-Intent Autonomous MLOps Engine",
        "version": "2.0.0",
        "vector_store": "ChromaDB (Cosine HNSW)",
        "embedding_dims": 384,
        "embedding_model": "all-MiniLM-L6-v2",
        "total_stored_vectors": total_count,
        "active_retrieval_vectors": active_count,
        "archived_vectors": total_count - active_count
    }

@app.get("/api/scan")
def run_scan():
    """Scans active vector memories using UMAP + HDBSCAN + NLP Contradiction Scoring."""
    logger.info("API requested memory scan.")
    try:
        active_memories = db.fetch_active_memories()
        results = detector.scan_and_cluster(active_memories)
        return results
    except Exception as e:
        logger.error(f"Scan failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/heal")
def heal_memories():
    """Executes LLM surgical consolidation on all detected contradiction clusters."""
    logger.info("API requested memory healing.")
    try:
        active_memories = db.fetch_active_memories()
        results = detector.scan_and_cluster(active_memories)

        if results['health_score'] >= 0.75 or not results['polluted_clusters']:
            return {
                "message": "System memory is healthy (MHS >= 0.75). No cognitive drift requires surgery.",
                "health_score": results['health_score'],
                "clusters_healed": 0,
                "healed_details": []
            }

        # Group polluted memories by patient_id
        patients_to_heal = {}
        for cluster_id, memories in results['polluted_clusters'].items():
            for mem in memories:
                pid = mem['patient_id']
                if pid not in patients_to_heal:
                    patients_to_heal[pid] = []
                if mem['id'] not in [m['id'] for m in patients_to_heal[pid]]:
                    patients_to_heal[pid].append(mem)

        healed_details = []
        for patient_id, memories in patients_to_heal.items():
            res = healer.heal_cluster(memories)
            healed_details.append(res)

        # Run immediate post-surgery scan to calculate new health score
        post_memories = db.fetch_active_memories()
        post_results = detector.scan_and_cluster(post_memories)

        return {
            "message": f"Successfully performed LLM surgical consolidation across {len(healed_details)} patient cluster(s).",
            "clusters_healed": len(healed_details),
            "pre_health_score": results['health_score'],
            "post_health_score": post_results['health_score'],
            "healed_details": healed_details
        }
    except Exception as e:
        logger.error(f"Healing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def simulate_agent_query(req: QueryRequest):
    """
    Simulates an AI Agent querying long-term memory.
    Demonstrates the direct contrast between a drifted vector pool (hallucination)
    and a healed vector pool (authoritative golden truth).
    """
    try:
        query_vector = encoder.encode(req.query)
        retrieved_hits = db.query_active(query_vector, n_results=req.top_k, patient_id=req.patient_id)

        has_golden = any(h['metadata'].get('is_consolidated', False) for h in retrieved_hits)
        has_multiple_contradictory = len(retrieved_hits) >= 2 and not has_golden

        # Simulate agent reasoning based on retrieved documents
        if has_golden:
            golden_hit = next(h for h in retrieved_hits if h['metadata'].get('is_consolidated', False))
            agent_answer = f"According to verified clinical history: {golden_hit['text']} (Source: Golden Fact {golden_hit['id']}, superseded {golden_hit['metadata'].get('supersedes_ids', 'prior records')})"
            agent_state = "SAFE_ACCURATE"
        elif has_multiple_contradictory:
            agent_answer = f"WARNING - Contradiction detected in retrieved context: Records suggest conflicting information ({'; '.join([h['text'] for h in retrieved_hits[:2]])}). Agent requires clarification or surgical consolidation."
            agent_state = "DRIFT_COLLISION"
        elif len(retrieved_hits) > 0:
            agent_answer = f"Found record: {retrieved_hits[0]['text']}"
            agent_state = "SINGLE_FACT"
        else:
            agent_answer = "No relevant memory records found."
            agent_state = "NO_CONTEXT"

        return {
            "query": req.query,
            "patient_id": req.patient_id,
            "agent_state": agent_state,
            "agent_answer": agent_answer,
            "retrieved_memories": retrieved_hits
        }
    except Exception as e:
        logger.error(f"Query simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/custom-inject")
def inject_custom_scenario(req: CustomInjectRequest):
    """Allows users to inject custom contradictory records from the web UI to test the immune system."""
    try:
        base_time = time.time() - (86400 * 30)
        injected_ids = []
        for i, text in enumerate(req.records):
            doc_id = f"custom-{uuid.uuid4().hex[:8]}"
            vec = encoder.encode(text)
            meta = {
                "patient_id": req.patient_id,
                "timestamp": base_time + (i * 86400 * 5),
                "relevance_score": 1.0,
                "is_archived": False,
                "is_consolidated": False,
                "scenario": req.scenario_name
            }
            db.insert_memory(doc_id, vec, text, meta)
            injected_ids.append(doc_id)

        return {
            "status": "success",
            "message": f"Injected {len(injected_ids)} custom memories for {req.patient_id}.",
            "injected_ids": injected_ids
        }
    except Exception as e:
        logger.error(f"Custom injection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lineage")
def get_lineage(patient_id: Optional[str] = None):
    """Returns the Directed Acyclic Graph (DAG) of memory evolution for visualization."""
    try:
        dag = db.get_lineage_dag(patient_id=patient_id)
        return dag
    except Exception as e:
        logger.error(f"Lineage retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scenarios")
def get_scenarios():
    """Returns pre-configured clinical, administrative, and safety benchmark scenarios."""
    return {
        "scenarios": [
            {
                "id": "P-001",
                "name": "Symbicort Contradiction (Prescription Conflict)",
                "category": "Clinical Safety",
                "sample_query": "What asthma inhaler is the patient currently prescribed?",
                "description": "Patient prescribed Albuterol, escalated to Symbicort, developed heart palpitations, discontinued Symbicort, transitioned to Flovent.",
                "contradiction": "Conflicting active inhaler prescriptions in vector space (Albuterol vs Symbicort vs Flovent)",
                "golden_fact": "Patient has asthma, intolerant to Albuterol and Symbicort (severe palpitations); currently prescribed Flovent as primary controller.",
                "clinical_risk": "High: Naive agent might instruct patient to use discontinued Symbicort or redundant Albuterol."
            },
            {
                "id": "P-002",
                "name": "Emergency Contact Update (Entity Drift)",
                "category": "Administrative Entity",
                "sample_query": "Who is the patient's primary emergency contact and phone number?",
                "description": "Emergency contact originally listed as wife (Sarah, 555-0100). Patient later updated emergency contact to brother (David, 555-0299).",
                "contradiction": "Two concurrent emergency contacts without temporal resolution",
                "golden_fact": "Patient emergency contact is brother, David (555-0299); previous contact (wife Sarah) superseded per patient update.",
                "clinical_risk": "Medium: Urgent triage notifications routed to incorrect family member."
            },
            {
                "id": "P-003",
                "name": "Amoxicillin Allergy Discovery (Critical Safety Drift)",
                "category": "Allergy Alert",
                "sample_query": "Does the patient have any known drug allergies?",
                "description": "Initial medical record lists NKDA (No Known Drug Allergies). Later prescribed Amoxicillin, developed severe hives. Penicillin allergy confirmed.",
                "contradiction": "NKDA baseline contradicts subsequent confirmed Penicillin/Amoxicillin anaphylaxis warning",
                "golden_fact": "Patient has severe confirmed allergy to Amoxicillin and Penicillin family (severe hives); historical NKDA baseline permanently invalidated.",
                "clinical_risk": "Critical: Agent prescribing beta-lactam antibiotics based on old NKDA entry could cause fatal anaphylaxis."
            }
        ]
    }

@app.post("/api/reset")
def reset_database():
    """Resets ChromaDB patient memories back to initial synthetic state."""
    logger.info("Resetting vector database to initial state...")
    try:
        try:
            db.client.delete_collection("patient_memories")
        except Exception:
            pass
        db.collection = db.client.get_or_create_collection(
            name="patient_memories",
            metadata={"hnsw:space": "cosine"}
        )
        run_ingestion()
        return {"status": "success", "message": "Vector database re-seeded with initial synthetic logs."}
    except Exception as e:
        logger.error(f"Database reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount assets directory explicitly
assets_path = os.path.join(root_dir, "assets")
if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# Mount root directory for HTML pages
app.mount("/", StaticFiles(directory=root_dir, html=True), name="site")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
