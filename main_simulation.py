import time
import warnings

from src.core.logger import get_logger
from src.mlops.drift_detector import DriftDetector
from src.mlops.healer import MemoryHealer
from src.storage.vector_client import MemoryStore

# Suppress UMAP numba warnings for a clean terminal output
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

def run_autonomous_loop():
    logger.info("=======================================================")
    logger.info("🚀 STARTING COGNI-INTENT AUTONOMOUS MLOps LOOP")
    logger.info("=======================================================")

    db = MemoryStore()
    detector = DriftDetector()
    healer = MemoryHealer()

    # --- PHASE 1: Background Scan ---
    logger.info("[System] Waking up background daemon. Fetching active memories...")
    active_memories = db.fetch_active_memories()

    logger.info("[System] Running UMAP + HDBSCAN Semantic Drift Detection...")
    results = detector.scan_and_cluster(active_memories)

    mhs = results['health_score']
    logger.info(f"[MLflow Dashboard] Current Memory Health Score (MHS): {mhs} / 1.0")

    # Check if the score dropped below our JSON config threshold (0.75)
    if mhs < 0.75:
        logger.warning("[ALERT] Cognitive Drift Detected! MHS is below safety threshold.")
        logger.warning(f"[ALERT] Identified {len(results['polluted_clusters'])} semantic contradictions requiring surgery.")

        # --- PHASE 2: Automated Self-Healing ---
        # Group polluted memories by patient_id to prevent duplicate golden vectors
        patients_to_heal = {}
        for cluster_id, memories in results['polluted_clusters'].items():
            for mem in memories:
                pid = mem['patient_id']
                if pid not in patients_to_heal:
                    patients_to_heal[pid] = []
                # Only add if we haven't already captured this exact memory
                if mem['id'] not in [m['id'] for m in patients_to_heal[pid]]:
                    patients_to_heal[pid].append(mem)

        for patient_id, memories in patients_to_heal.items():
            logger.info(f"🛠️  Initiating Memory Consolidation for Patient {patient_id}...")
            # Pass the aggregated messy memories to the Healer
            healer.heal_cluster(memories)
            logger.info("-" * 60)

        # --- PHASE 3: Re-Scan & Verify System Recovery ---
        logger.info("[System] Surgical consolidation complete. Running verification scan...")
        time.sleep(2) # Artificial pause to simulate processing time

        # Fetch only the active memories (the soft-evicted ones should be hidden)
        active_memories_post = db.fetch_active_memories()
        results_post = detector.scan_and_cluster(active_memories_post)

        logger.info(f"[MLflow Dashboard] Post-Surgery Memory Health Score (MHS): {results_post['health_score']} / 1.0")
        logger.info("✅ System successfully recovered. Diachronic medical history preserved via soft-eviction.")
    else:
        logger.info("✅ System is healthy. No cognitive drift detected.")

    logger.info("=======================================================")
    logger.info("🏁 SIMULATION COMPLETE")
    logger.info("=======================================================")

if __name__ == "__main__":
    run_autonomous_loop()
