import uuid
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from colony.training.roles import ROLE_PROMPTS


class NeuronState(Enum):
    DORMANT = "dormant"
    FIRING = "firing"
    REFRACTORY = "refractory"
    PRUNED = "pruned"


@dataclass
class NeuronAgent:
    """
    A single neuron in the cortex. Competes for survival via usefulness.
    Connections to other neurons strengthen/weaken via Hebbian learning.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: Optional[str] = None             # analyst | critic | synthesizer | explorer
    adapter_path: Optional[str] = None     # resolved from role by cortex
    specialization: Optional[str] = None  # fine-grained label, emerges over time

    survival_score: float = 0.5
    fire_count: int = 0
    success_count: int = 0
    last_fired: float = field(default_factory=time.time)

    x: float = 0.0
    y: float = 0.0

    state: NeuronState = NeuronState.DORMANT

    @property
    def fitness(self) -> float:
        if self.fire_count == 0:
            return 0.0
        return self.success_count / self.fire_count

    def build_prompt(self, task: str, context: str = "", examples: list[str] | None = None) -> str:
        role_line = ROLE_PROMPTS.get(self.role, "") if self.role else ""
        if self.specialization and not role_line:
            role_line = f"You are specialized in: {self.specialization}."
        ctx_line = f"\nContext from connected neurons:\n{context}\n" if context else ""
        ex_line = ""
        if examples:
            ex_line = "\nPast successful responses on similar tasks:\n"
            for i, ex in enumerate(examples, 1):
                ex_line += f"[{i}] {ex[:200]}\n"
            ex_line += "\n"
        system = role_line + ctx_line + ex_line
        return f"{system}\nTask: {task}\n\nResponse:" if system else f"Task: {task}\n\nResponse:"

    def update_survival(self, success: bool, per_fire_decay: float = 0.01):
        """
        Increase survival on success, penalise on failure.
        per_fire_decay is applied every fire regardless of outcome —
        neurons must keep contributing to survive.
        """
        self.fire_count += 1
        if success:
            self.success_count += 1
            self.survival_score = min(1.0, self.survival_score + 0.1)
        else:
            self.survival_score = max(0.0, self.survival_score - 0.05)

        self.survival_score = max(0.0, self.survival_score - per_fire_decay)
        self.last_fired = time.time()

    def __repr__(self):
        return f"Neuron({self.id}, score={self.survival_score:.2f}, fires={self.fire_count})"
