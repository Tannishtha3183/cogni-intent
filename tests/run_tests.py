import os
import sys

# Add root directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from src.ingestion.encoder import TextEncoder
from src.mlops.drift_detector import DriftDetector
from src.mlops.healer import MemoryHealer
from src.storage.vector_client import MemoryStore


def run_suite():
    print("=== Cogni-Intent Test Suite ===")

    # 1. TextEncoder test
    print("[Test 1] Testing TextEncoder...")
    encoder = TextEncoder()
    vec = encoder.encode("Patient reports mild asthma symptoms.")
    assert len(vec) == 384, f"Expected 384 dims, got {len(vec)}"
    print(" -> PASS: TextEncoder generates valid 384-D vector.")

    # 2. VectorStore test
    print("[Test 2] Testing MemoryStore...")
    test_db_path = "data/test_chroma_db"
    if os.path.exists(test_db_path):
        import shutil
        shutil.rmtree(test_db_path, ignore_errors=True)
    db = MemoryStore(db_path=test_db_path)
    doc_id = "test-doc-001"
    db.insert_memory(
        doc_id=doc_id,
        vector=vec,
        text="Patient reports mild asthma symptoms.",
        metadata={"patient_id": "P-TEST", "is_archived": False, "is_consolidated": False}
    )
    active = db.fetch_active_memories()
    assert doc_id in active['ids'], "Inserted doc should be in active memories"

    db.soft_delete(doc_id)
    active_after = db.fetch_active_memories()
    assert doc_id not in active_after['ids'], "Soft-deleted doc should NOT be in active memories"

    all_docs = db.fetch_all()
    assert doc_id in all_docs['ids'], "Soft-deleted doc MUST be in fetch_all (audit trail)"
    print(" -> PASS: MemoryStore soft-eviction and active filtering work as specified.")

    # 3. DriftDetector test
    print("[Test 3] Testing DriftDetector...")
    detector = DriftDetector()
    sample_memories = {
        "embeddings": [vec, encoder.encode("Patient prescribed Symbicort.")],
        "ids": ["doc-1", "doc-2"],
        "documents": ["Patient reports mild asthma symptoms.", "Patient prescribed Symbicort."],
        "metadatas": [{"patient_id": "P-001"}, {"patient_id": "P-001"}]
    }
    scan_res = detector.scan_and_cluster(sample_memories)
    assert "health_score" in scan_res, "Scan response must include health_score"
    assert "points" in scan_res, "Scan response must include 2D visualization points"
    print(f" -> PASS: DriftDetector generated MHS={scan_res['health_score']}, severity={scan_res['severity']}.")

    # 4. MemoryHealer test
    print("[Test 4] Testing MemoryHealer...")
    healer = MemoryHealer()
    cluster_to_heal = [
        {"id": "doc-1", "text": "Prescribed Symbicort.", "patient_id": "P-001"},
        {"id": "doc-2", "text": "Symbicort discontinued due to palpitations. Switched to Flovent.", "patient_id": "P-001"}
    ]
    heal_res = healer.heal_cluster(cluster_to_heal)
    assert "golden_fact" in heal_res, "Heal response must contain golden_fact"
    assert "superseded_ids" in heal_res, "Heal response must track superseded IDs"
    print(f" -> PASS: Healer generated Golden Fact: '{heal_res['golden_fact'][:50]}...'")

    print("\n✅ ALL BACKEND TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_suite()
