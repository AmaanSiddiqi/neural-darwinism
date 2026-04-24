import pytest
from colony.agents.neuron import NeuronAgent, NeuronState


def test_fitness_zero_fires():
    n = NeuronAgent()
    assert n.fitness == 0.0


def test_fitness_all_success():
    n = NeuronAgent()
    n.update_survival(True)
    n.update_survival(True)
    assert n.fitness == 1.0


def test_fitness_partial():
    n = NeuronAgent()
    n.update_survival(True)
    n.update_survival(False)
    assert n.fitness == pytest.approx(0.5)


def test_survival_increases_on_success():
    n = NeuronAgent(survival_score=0.5)
    n.update_survival(True)
    assert n.survival_score > 0.5


def test_survival_decreases_on_failure():
    n = NeuronAgent(survival_score=0.5)
    n.update_survival(False)
    assert n.survival_score < 0.5


def test_survival_clamps_at_zero():
    n = NeuronAgent(survival_score=0.0)
    n.update_survival(False)
    assert n.survival_score == 0.0


def test_survival_clamps_at_one():
    n = NeuronAgent(survival_score=1.0)
    # Success tries to add 0.1 but should clamp at 1.0, then decay by 0.01
    n.update_survival(True, per_fire_decay=0.0)
    assert n.survival_score == 1.0


def test_build_prompt_no_specialization():
    n = NeuronAgent()
    prompt = n.build_prompt("Solve X")
    assert "Task: Solve X" in prompt
    assert "specialized" not in prompt


def test_build_prompt_with_specialization():
    n = NeuronAgent(specialization="mathematics")
    prompt = n.build_prompt("Solve X")
    assert "mathematics" in prompt


def test_build_prompt_with_context():
    n = NeuronAgent()
    prompt = n.build_prompt("Solve X", context="prior answer")
    assert "prior answer" in prompt


def test_unique_ids():
    ids = {NeuronAgent().id for _ in range(100)}
    assert len(ids) == 100
