"""
FastAPI server — serves the D3 visualization and streams cortex events via WebSocket.
"""
import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import colony.config as cfg

app = FastAPI(title="Colony")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state — one colony runs at a time
_model = None
_cortex = None
_episodic = None
_role_memory = None
_run_task: Optional[asyncio.Task] = None
_active_ws: list[WebSocket] = []
_state = {"running": False, "generation": 0, "total": 0}
_adapter_versions: dict[str, int] = {}
_benchmark_history: list[dict] = []  # [{label, mean_score, tasks}]

def _load_benchmark_history():
    global _benchmark_history
    p = Path(cfg.BENCHMARK_HISTORY_PATH)
    if p.exists():
        try:
            _benchmark_history = json.loads(p.read_text())
        except Exception:
            _benchmark_history = []

def _save_benchmark_history():
    Path(cfg.BENCHMARK_HISTORY_PATH).write_text(json.dumps(_benchmark_history, indent=2))

_load_benchmark_history()


class RunConfig(BaseModel):
    task: str = "What are the most important second-order consequences of AI becoming widely available?"
    generations: int = 200
    delay_ms: int = 1000
    use_model: bool = True
    seed_n: int = 8
    resume: bool = True
    benchmark_interval: int = cfg.BENCHMARK_INTERVAL


def _load_model():
    """Synchronous — must be called via run_in_executor."""
    global _model
    if _model is None:
        from colony.models.model_manager import ModelManager
        _model = ModelManager()
    return _model


def _init_memory():
    global _episodic, _role_memory, _adapter_versions
    if _episodic is None:
        from colony.memory import EpisodicMemory, RoleMemory
        _episodic = EpisodicMemory(persist_dir=cfg.CHROMA_DIR)
        _role_memory = RoleMemory.load(
            cfg.ROLE_MEMORY_PATH,
            capacity=cfg.MEMORY_CAPACITY,
            score_threshold=cfg.MEMORY_SCORE_THRESHOLD,
        )
        # Restore adapter version counters from persisted memory
        _adapter_versions = dict(_role_memory._versions)


def _cortex_state(cortex) -> dict:
    neurons = []
    for n in cortex.neurons.values():
        neurons.append({
            "id": n.id,
            "role": n.role or "unknown",
            "survival": round(n.survival_score, 3),
            "fitness": round(n.fitness, 3),
            "fires": n.fire_count,
            "state": n.state.value,
            "adapter": n.adapter_path is not None,
            "x": round(n.x, 4),
            "y": round(n.y, 4),
        })
    edges = []
    for src, dst, data in cortex.graph.edges(data=True):
        if src in cortex.neurons and dst in cortex.neurons:
            edges.append({"src": src, "dst": dst, "weight": round(data.get("weight", 0.1), 3)})
    return {"neurons": neurons, "edges": edges}


async def _broadcast(msg: dict):
    dead = []
    for ws in _active_ws:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _active_ws:
            _active_ws.remove(ws)


def _write_memories(result: dict, task: str, cortex):
    """Write winning responses to both memory systems."""
    if not _role_memory or not _episodic:
        return []
    ready_roles = []
    for r in result.get("results", []):
        if not r["success"]:
            continue
        n = cortex.neurons.get(r["id"])
        if not n or not n.role:
            continue
        score = r["score"]
        hit_capacity = _role_memory.add(n.role, task, r["response"], score)
        _episodic.store(n.role, task, r["response"], score)
        if hit_capacity:
            ready_roles.append(n.role)
    return ready_roles


async def _run_colony(config: RunConfig):
    global _cortex, _state, _adapter_versions
    from colony.graph.cortex import Cortex

    _state["running"] = True
    _state["generation"] = 0
    _state["total"] = config.generations

    loop = asyncio.get_running_loop()

    await loop.run_in_executor(None, _init_memory)

    if config.use_model:
        await _broadcast({"type": "loading", "message": "Loading model..."})
        model = await loop.run_in_executor(None, _load_model)
    else:
        model = None

    state_path = Path(cfg.CORTEX_STATE_PATH)
    resumed_from = 0
    if config.resume and state_path.exists():
        _cortex = Cortex.load(str(state_path), model_manager=model, adapter_dir=cfg.ADAPTER_DIR, episodic=_episodic)
        resumed_from = _cortex.generation
    else:
        _cortex = Cortex(model_manager=model, episodic=_episodic, adapter_dir=cfg.ADAPTER_DIR).seed(n=config.seed_n)

    await _broadcast({
        "type": "init",
        **_cortex_state(_cortex),
        "total": config.generations,
        "resumed_from": resumed_from,
    })

    try:
        for gen in range(1, config.generations + 1):
            _state["generation"] = gen

            # Fine-tune any roles that hit memory capacity last generation
            if model and _role_memory:
                for role in _role_memory.pop_ready():
                    entries = _role_memory.drain(role)
                    if entries:
                        ver = _adapter_versions.get(role, 0)
                        await _broadcast({
                            "type": "training_start",
                            "role": role,
                            "examples": len(entries),
                            "from_version": ver,
                        })
                        await loop.run_in_executor(None, model.fine_tune_role, role, entries)
                        new_ver = _role_memory.bump_version(role)
                        _adapter_versions[role] = new_ver
                        await loop.run_in_executor(None, _role_memory.save, cfg.ROLE_MEMORY_PATH)
                        await _broadcast({
                            "type": "adapter_ready",
                            "role": role,
                            "version": new_ver,
                        })

            result = await loop.run_in_executor(None, _cortex.step, config.task)

            # Write winners to memory banks (synchronous — fast)
            _write_memories(result, config.task, _cortex)

            events = (
                [{"type": "pruned", "id": nid} for nid in result.get("pruned", [])]
                + [{"type": "born",   "id": nid} for nid in result.get("born",   [])]
            )

            role_score_map: dict = {}
            for r in result.get("results", []):
                n = _cortex.neurons.get(r["id"])
                if n and n.role:
                    role_score_map.setdefault(n.role, []).append(r["score"])
            role_avg_scores = {
                role: round(sum(v) / len(v), 3)
                for role, v in role_score_map.items()
            }

            survivals = [n.survival_score for n in _cortex.neurons.values()]
            survival_stats = {
                "mean": round(sum(survivals) / len(survivals), 3),
                "min":  round(min(survivals), 3),
                "max":  round(max(survivals), 3),
            } if survivals else {"mean": 0, "min": 0, "max": 0}

            mem_counts = _role_memory.counts() if _role_memory else {}

            await _broadcast({
                "type":                "generation",
                "generation":          gen,
                "total":               config.generations,
                **_cortex_state(_cortex),
                "events":              events,
                "best_response":       result.get("best_response", ""),
                "best_score":          round(result.get("best_score", 0), 3),
                "scores": [
                    {"id": r["id"], "score": round(r["score"], 3), "success": r["success"]}
                    for r in result.get("results", [])
                ],
                "neuron_count":        result["neuron_count"],
                "role_scores":         role_avg_scores,
                "survival_stats":      survival_stats,
                "synthesized_response": result.get("synthesized_response", ""),
                "synthesizer_id":      result.get("synthesizer_id", ""),
                "memory": {
                    "episodic_count": _episodic.count() if _episodic else 0,
                    "banks":          mem_counts,
                    "capacity":       cfg.MEMORY_CAPACITY,
                    "versions":       dict(_adapter_versions),
                },
            })

            # Periodic save every 10 generations
            if gen % 10 == 0:
                await loop.run_in_executor(None, _cortex.save, str(state_path))
                if _role_memory:
                    await loop.run_in_executor(None, _role_memory.save, cfg.ROLE_MEMORY_PATH)

            # Auto-benchmark at configured interval
            if config.benchmark_interval > 0 and gen % config.benchmark_interval == 0 and model:
                def _auto_benchmark():
                    from colony.benchmark import run_benchmark
                    return run_benchmark(lambda p: model.generate(p, role=None, max_new_tokens=150))
                bm = await loop.run_in_executor(None, _auto_benchmark)
                label = f"gen {_cortex.generation}"
                entry = {"label": label, **bm}
                _benchmark_history.append(entry)
                _save_benchmark_history()
                await _broadcast({"type": "benchmark", **entry, "history": _benchmark_history})

            if not _active_ws:
                break

            await asyncio.sleep(config.delay_ms / 1000)

        await _broadcast({"type": "done", "generation": config.generations})
    except asyncio.CancelledError:
        await _broadcast({"type": "stopped"})
    finally:
        _state["running"] = False
        if _cortex:
            _cortex.save(str(state_path))
        if _role_memory:
            _role_memory.save(cfg.ROLE_MEMORY_PATH)


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/status")
async def status():
    return _state


@app.post("/run")
async def start_run(config: RunConfig):
    global _run_task
    if _run_task and not _run_task.done():
        _run_task.cancel()
        await asyncio.sleep(0.1)
    _run_task = asyncio.create_task(_run_colony(config))
    return {"status": "started"}


@app.post("/stop")
async def stop_run():
    global _run_task
    if _run_task and not _run_task.done():
        _run_task.cancel()
    return {"status": "stopped"}


@app.post("/benchmark")
async def run_benchmark_endpoint():
    """Score holdout tasks with current model weights. Call before and after training."""
    global _benchmark_history
    if not _model:
        return {"error": "Model not loaded — run the colony first"}

    loop = asyncio.get_running_loop()

    def _do_benchmark():
        from colony.benchmark import run_benchmark
        return run_benchmark(lambda p: _model.generate(p, role=None, max_new_tokens=150))

    result = await loop.run_in_executor(None, _do_benchmark)
    label = f"gen {_state['generation']}"
    entry = {"label": label, **result}
    _benchmark_history.append(entry)
    _save_benchmark_history()
    await _broadcast({"type": "benchmark", **entry, "history": _benchmark_history})
    return entry


@app.get("/benchmark/history")
async def benchmark_history():
    return _benchmark_history


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _active_ws.append(ws)
    await ws.send_json({"type": "status", **_state})
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        if ws in _active_ws:
            _active_ws.remove(ws)
