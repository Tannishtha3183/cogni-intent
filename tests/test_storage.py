import pytest
import os
from src.storage.vector_client import MemoryStore

@pytest.fixture
def memory_store(tmp_path):
    db_path = str(tmp_path / "chroma_test_db")
    return MemoryStore(db_path=db_path)

def test_insert_and_fetch_active_memories(memory_store):
    memory_store.insert_memory(
        doc_id="test-1",
        vector=[0.1] * 384,
        text="Test memory",
        metadata={"patient_id": "P-TEST", "is_archived": False}
    )
    active = memory_store.fetch_active_memories()
    assert len(active['ids']) == 1
    assert active['ids'][0] == "test-1"

def test_soft_delete(memory_store):
    memory_store.insert_memory(
        doc_id="test-2",
        vector=[0.2] * 384,
        text="Test memory 2",
        metadata={"patient_id": "P-TEST", "is_archived": False}
    )
    memory_store.soft_delete("test-2")
    active = memory_store.fetch_active_memories()
    assert "test-2" not in active['ids']
    all_mems = memory_store.fetch_all()
    assert "test-2" in all_mems['ids']
