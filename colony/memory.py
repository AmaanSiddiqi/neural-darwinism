import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class MemoryEntry:
    task: str
    response: str
    score: float
    role: str


class RoleMemory:
    """
    Per-role bank of winning responses for LoRA fine-tuning.
    When a role accumulates CAPACITY high-scoring examples, it signals ready.
    """

    def __init__(self, capacity: int = 16, score_threshold: float = 0.65):
        self.capacity = capacity
        self.score_threshold = score_threshold
        self._banks: dict[str, list[MemoryEntry]] = {}
        self._versions: dict[str, int] = {}
        self._ready: set[str] = set()
        self._lock = threading.Lock()

    def add(self, role: str, task: str, response: str, score: float) -> bool:
        """Store entry if score meets threshold. Returns True when role hits capacity."""
        if score < self.score_threshold:
            return False
        with self._lock:
            bank = self._banks.setdefault(role, [])
            bank.append(MemoryEntry(task, response, score, role))
            if len(bank) >= self.capacity and role not in self._ready:
                self._ready.add(role)
                return True
        return False

    def pop_ready(self) -> list[str]:
        """Return and clear roles that have hit training capacity."""
        with self._lock:
            ready = list(self._ready)
            self._ready.clear()
            return ready

    def drain(self, role: str) -> list[MemoryEntry]:
        """Remove and return all entries for a role."""
        with self._lock:
            return self._banks.pop(role, [])

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {r: len(b) for r, b in self._banks.items()}

    def version(self, role: str) -> int:
        return self._versions.get(role, 0)

    def bump_version(self, role: str) -> int:
        self._versions[role] = self._versions.get(role, 0) + 1
        return self._versions[role]

    def save(self, path: str):
        with self._lock:
            data = {
                "banks": {
                    role: [{"task": e.task, "response": e.response, "score": e.score, "role": e.role}
                           for e in entries]
                    for role, entries in self._banks.items()
                },
                "versions": dict(self._versions),
            }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str, capacity: int = 16, score_threshold: float = 0.65) -> "RoleMemory":
        mem = cls(capacity=capacity, score_threshold=score_threshold)
        p = Path(path)
        if not p.exists():
            return mem
        try:
            data = json.loads(p.read_text())
            with mem._lock:
                for role, entries in data.get("banks", {}).items():
                    mem._banks[role] = [
                        MemoryEntry(task=e["task"], response=e["response"], score=e["score"], role=e["role"])
                        for e in entries
                    ]
                    if len(mem._banks[role]) >= mem.capacity:
                        mem._ready.add(role)
                mem._versions = data.get("versions", {})
        except Exception as e:
            print(f"[RoleMemory] Failed to load {path}: {e}")
        return mem


class EpisodicMemory:
    """
    Vector store of past winning responses.
    Neurons query this before generating — retrieval-augmented generation
    from the colony's own successful history (episodic memory).
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="colony_wins",
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, role: str, task: str, response: str, score: float):
        self._collection.add(
            ids=[str(uuid.uuid4())[:12]],
            documents=[response],
            metadatas=[{"role": role, "task": task[:200], "score": score}],
        )

    def retrieve(self, task: str, role: str, top_k: int = 2) -> list[str]:
        """Return top-k past winning responses for similar tasks and same role."""
        if not role:
            return []
        n = self._collection.count()
        if n == 0:
            return []
        try:
            results = self._collection.query(
                query_texts=[task],
                n_results=min(top_k, n),
                where={"role": role},
            )
            return [d for d in (results["documents"][0] if results["documents"] else []) if d]
        except Exception:
            return []

    def count(self) -> int:
        return self._collection.count()
