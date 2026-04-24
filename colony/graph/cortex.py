import random
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import networkx as nx
from typing import Optional
from rich.console import Console
from rich.table import Table

# Persistent thread pool — avoids per-step thread creation overhead.
# max_workers=4 matches the active neuron count per generation.
_judge_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="colony-judge")

import colony.config as cfg
from colony.agents.neuron import NeuronAgent, NeuronState
from colony.training.roles import ROLES

console = Console()

ROLE_COLORS = {
    "analyst": "blue",
    "critic": "red",
    "synthesizer": "green",
    "explorer": "magenta",
}


class Cortex:
    """
    The neural graph. Manages neuron lifecycle and Hebbian dynamics:
      - Hebbian learning: co-successful neurons strengthen connections
      - Pruning: neurons below survival threshold are removed
      - Neurogenesis: new neurons spawn near thriving clusters
    """

    def __init__(self, model_manager=None, adapter_dir: str = "./adapters", episodic=None):
        self.graph = nx.DiGraph()
        self.neurons: dict[str, NeuronAgent] = {}
        self.model = model_manager
        self.episodic = episodic
        self.generation = 0
        self.adapter_dir = Path(adapter_dir)

    def _resolve_adapter(self, neuron: NeuronAgent):
        """Mark whether a trained adapter exists for this role."""
        if neuron.role:
            neuron.adapter_path = str(self.adapter_dir / neuron.role) if (self.adapter_dir / neuron.role).exists() else None

    def _assign_role(self, neuron: NeuronAgent, parent: Optional[NeuronAgent] = None):
        """Assign role: inherit from parent with 20% mutation chance, else random."""
        if parent and parent.role and random.random() > 0.2:
            neuron.role = parent.role
        else:
            neuron.role = random.choice(ROLES)
        self._resolve_adapter(neuron)

    def add_neuron(self, neuron: Optional[NeuronAgent] = None, connect_to: Optional[list[str]] = None) -> NeuronAgent:
        if neuron is None:
            neuron = NeuronAgent()

        parent = self.neurons[connect_to[0]] if connect_to else None
        if parent:
            neuron.x = parent.x + random.uniform(-0.3, 0.3)
            neuron.y = parent.y + random.uniform(-0.3, 0.3)
        else:
            neuron.x = random.uniform(-1, 1)
            neuron.y = random.uniform(-1, 1)

        # Assign role if not already set
        if neuron.role is None:
            self._assign_role(neuron, parent=parent)

        existing_ids = list(self.neurons.keys())
        self.neurons[neuron.id] = neuron
        self.graph.add_node(neuron.id, survival=neuron.survival_score)

        candidates = connect_to or random.sample(existing_ids, k=min(2, len(existing_ids)))
        for target_id in candidates:
            if target_id in self.neurons:
                self.graph.add_edge(neuron.id, target_id, weight=0.5)
                self.graph.add_edge(target_id, neuron.id, weight=0.5)

        return neuron

    def seed(self, n: int = 8) -> "Cortex":
        for _ in range(n):
            self.add_neuron()
        return self

    def get_neighbors(self, neuron_id: str, top_k: int = 3) -> list[NeuronAgent]:
        if neuron_id not in self.graph:
            return []
        edges = sorted(
            self.graph.out_edges(neuron_id, data=True),
            key=lambda e: e[2].get("weight", 0),
            reverse=True,
        )
        return [self.neurons[n] for _, n, _ in edges[:top_k] if n in self.neurons]

    def hebbian_update(self, successful_ids: list[str], failed_ids: list[str]):
        """
        Strengthen connections between neurons that both succeeded.
        Weaken connections that include a failing neuron.
        """
        # Strengthen pairs that co-succeeded
        for src in successful_ids:
            for dst in successful_ids:
                if src == dst:
                    continue
                if self.graph.has_edge(src, dst):
                    old = self.graph[src][dst]["weight"]
                    self.graph[src][dst]["weight"] = min(1.0, old + cfg.HEBBIAN_LR)
                else:
                    self.graph.add_edge(src, dst, weight=cfg.HEBBIAN_LR)

        # Weaken connections that cross a failure boundary
        all_ids = successful_ids + failed_ids
        for src in failed_ids:
            for dst in all_ids:
                if src == dst:
                    continue
                if self.graph.has_edge(src, dst):
                    old = self.graph[src][dst]["weight"]
                    self.graph[src][dst]["weight"] = max(0.01, old - cfg.HEBBIAN_LR * 0.5)

    def prune(self) -> list[str]:
        dead = [nid for nid, n in self.neurons.items() if n.survival_score < cfg.PRUNE_THRESHOLD]
        for nid in dead:
            self.neurons[nid].state = NeuronState.PRUNED
            self.graph.remove_node(nid)
            del self.neurons[nid]
        if dead:
            console.print(f"[red]Pruned {len(dead)} neuron(s): {dead}[/red]")
        return dead

    def neurogenesis(self) -> list[NeuronAgent]:
        if len(self.neurons) >= cfg.MAX_NEURONS:
            return []

        thriving = [n for n in self.neurons.values() if n.survival_score >= cfg.NEUROGENESIS_THRESHOLD]
        if not thriving:
            return []

        spawned = []
        for parent in thriving:
            if random.random() < 0.4 and len(self.neurons) < cfg.MAX_NEURONS:
                child = NeuronAgent(survival_score=0.4)
                self.add_neuron(child, connect_to=[parent.id])
                spawned.append(child)
                console.print(f"[green]Neurogenesis: {child.id} spawned from {parent.id}[/green]")

        return spawned

    def decay_all(self):
        for neuron in self.neurons.values():
            neuron.survival_score = max(0.0, neuron.survival_score - 0.005)

    def synthesize(self, task: str, top_responses: list[dict]) -> tuple[str, str]:
        """
        Combine the top survivors' perspectives into one unified answer.
        Prefers a synthesizer-role neuron; falls back to highest-survival neuron.
        Returns (synthesized_text, synthesizer_id).
        """
        if not self.model or len(top_responses) < 2:
            return "", ""

        synthesizer = next(
            (n for n in sorted(self.neurons.values(), key=lambda n: n.survival_score, reverse=True)
             if n.role == "synthesizer"),
            None,
        ) or max(self.neurons.values(), key=lambda n: n.survival_score, default=None)

        if not synthesizer:
            return "", ""

        perspectives = "\n\n".join(
            f"[{self.neurons.get(r['id'], NeuronAgent()).role or 'agent'}]: {r['response'][:300]}"
            for r in top_responses
        )
        prompt = (
            f"Synthesize the following agent perspectives into one high-quality answer.\n"
            f"Task: {task}\n\n"
            f"{perspectives}\n\n"
            f"Unified response:"
        )

        result = self.model.generate(prompt, role="synthesizer", max_new_tokens=200)
        synthesizer.survival_score = min(1.0, synthesizer.survival_score + 0.05)
        return result, synthesizer.id

    def step(self, task: str) -> dict:
        """
        One cortex cycle:
          1. Activate a random subset of neurons
          2. Each generates a response
          3. Score and rank responses; bottom half fail (competitive selection)
          4. Hebbian update between co-successful neurons
          5. Prune dying neurons, run neurogenesis
          6. Synthesis: top survivors combine perspectives into a final answer
        """
        self.generation += 1
        self.decay_all()

        if not self.neurons:
            return {"error": "No neurons", "generation": self.generation}

        active = random.sample(list(self.neurons.values()), k=min(4, len(self.neurons)))
        for n in active:
            n.state = NeuronState.FIRING

        # Generate + pipeline judge:
        # After each generate() (GPU), immediately submit the judge API call to the
        # thread pool. It runs concurrently while the next neuron is generating,
        # hiding API latency behind GPU compute. All judge calls also run in parallel
        # with each other — 4 API calls take ~0.5s instead of ~2s sequential.
        responses: list[dict] = []
        futures: list[Future | None] = []

        for neuron in active:
            context = ""
            neighbors = self.get_neighbors(neuron.id)
            if neighbors and responses:
                prev = next((r for r in responses if r["id"] in {nb.id for nb in neighbors}), None)
                if prev:
                    context = prev["response"][:300]

            examples = self.episodic.retrieve(task, neuron.role) if self.episodic else []

            if self.model:
                response = self.model.generate(neuron.build_prompt(task, context, examples), role=neuron.role)
                futures.append(_judge_pool.submit(self.model.judge, task, response))
            else:
                response = f"[mock] Neuron {neuron.id} processed: {task[:60]}"
                futures.append(None)

            responses.append({"id": neuron.id, "response": response, "neuron": neuron, "score": 0.0})

        # Collect scores — most futures already resolved while we were generating
        for r, future in zip(responses, futures):
            if future is not None:
                _, r["score"] = future.result()
            else:
                r["score"] = float(len(r["response"]) > 20)

        # Competitive selection: top half pass
        ranked = sorted(responses, key=lambda r: r["score"], reverse=True)
        cutoff = max(1, len(ranked) // 2)
        passing_ids = {r["id"] for r in ranked[:cutoff]}

        results = []
        successful_ids, failed_ids = [], []
        for r in responses:
            success = r["id"] in passing_ids
            r["neuron"].update_survival(success)
            r["neuron"].state = NeuronState.REFRACTORY
            (successful_ids if success else failed_ids).append(r["id"])
            results.append({"id": r["id"], "response": r["response"], "score": r["score"], "success": success})

        self.hebbian_update(successful_ids, failed_ids)

        pruned = self.prune()
        born = self.neurogenesis()

        best = max(results, key=lambda r: r["score"])

        # Synthesis: surviving neurons combine their perspectives
        top = ranked[:cutoff]
        synthesized, synthesizer_id = self.synthesize(task, top)

        return {
            "generation": self.generation,
            "active": len(active),
            "results": results,
            "best_response": best["response"],
            "best_score": best["score"],
            "synthesized_response": synthesized,
            "synthesizer_id": synthesizer_id,
            "pruned": pruned,
            "born": [n.id for n in born],
            "neuron_count": len(self.neurons),
        }

    def status(self):
        table = Table(title=f"Cortex — Generation {self.generation}")
        table.add_column("ID", style="cyan")
        table.add_column("Role")
        table.add_column("Survival", style="magenta")
        table.add_column("Fires")
        table.add_column("Fitness", style="green")
        table.add_column("Adapter")
        table.add_column("State")

        for n in sorted(self.neurons.values(), key=lambda x: x.survival_score, reverse=True):
            role_color = ROLE_COLORS.get(n.role, "white")
            role_str = f"[{role_color}]{n.role or '—'}[/{role_color}]"
            adapter_str = "[green]yes[/green]" if n.adapter_path else "[dim]no[/dim]"
            table.add_row(
                n.id,
                role_str,
                f"{n.survival_score:.3f}",
                str(n.fire_count),
                f"{n.fitness:.2f}",
                adapter_str,
                n.state.value,
            )
        console.print(table)

    def save(self, path: str):
        """Persist neuron states, Hebbian edge weights, and generation count to JSON."""
        import json
        Path(path).write_text(json.dumps({
            "generation": self.generation,
            "neurons": [
                {
                    "id": n.id,
                    "role": n.role,
                    "survival_score": n.survival_score,
                    "fire_count": n.fire_count,
                    "success_count": n.success_count,
                    "x": n.x,
                    "y": n.y,
                    "adapter_path": n.adapter_path,
                }
                for n in self.neurons.values()
            ],
            "edges": [
                {"src": src, "dst": dst, "weight": data.get("weight", 0.5)}
                for src, dst, data in self.graph.edges(data=True)
                if src in self.neurons and dst in self.neurons
            ],
        }, indent=2))

    @classmethod
    def load(cls, path: str, model_manager=None, adapter_dir: str = "./adapters", episodic=None) -> "Cortex":
        """Restore a cortex from a saved JSON state."""
        import json
        state = json.loads(Path(path).read_text())
        cortex = cls(model_manager=model_manager, adapter_dir=adapter_dir, episodic=episodic)
        cortex.generation = state["generation"]

        for nd in state["neurons"]:
            neuron = NeuronAgent(
                id=nd["id"],
                role=nd["role"],
                survival_score=nd["survival_score"],
                fire_count=nd["fire_count"],
                success_count=nd["success_count"],
                x=nd["x"],
                y=nd["y"],
                adapter_path=nd.get("adapter_path"),
            )
            cortex.neurons[neuron.id] = neuron
            cortex.graph.add_node(neuron.id, survival=neuron.survival_score)

        for ed in state["edges"]:
            if ed["src"] in cortex.neurons and ed["dst"] in cortex.neurons:
                cortex.graph.add_edge(ed["src"], ed["dst"], weight=ed["weight"])

        print(f"[Cortex] Resumed from generation {cortex.generation} "
              f"({len(cortex.neurons)} neurons, {cortex.graph.number_of_edges()} edges)")
        return cortex

    def role_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {r: 0 for r in ROLES}
        for n in self.neurons.values():
            if n.role and n.role in dist:
                dist[n.role] += 1
        return dist
