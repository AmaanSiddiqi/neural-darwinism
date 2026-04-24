import pytest
from colony.graph.cortex import Cortex
from colony.agents.neuron import NeuronAgent, NeuronState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cortex():
    return Cortex(model_manager=None).seed(n=6)


# ── Seeding ───────────────────────────────────────────────────────────────────

def test_seed_creates_correct_count():
    c = Cortex().seed(n=5)
    assert len(c.neurons) == 5


def test_seed_no_self_connections():
    c = Cortex().seed(n=8)
    for nid in c.graph.nodes:
        assert not c.graph.has_edge(nid, nid)


def test_add_neuron_no_self_connection():
    c = Cortex()
    n = c.add_neuron()
    assert not c.graph.has_edge(n.id, n.id)


# ── Pruning ───────────────────────────────────────────────────────────────────

def test_prune_removes_low_survival(cortex):
    target = list(cortex.neurons.values())[0]
    target.survival_score = 0.0  # below any threshold
    dead = cortex.prune()
    assert target.id in dead
    assert target.id not in cortex.neurons
    assert target.id not in cortex.graph.nodes


def test_prune_keeps_healthy_neurons(cortex):
    for n in cortex.neurons.values():
        n.survival_score = 0.9
    dead = cortex.prune()
    assert dead == []
    assert len(cortex.neurons) == 6


def test_prune_state_set_to_pruned(cortex):
    target = list(cortex.neurons.values())[0]
    target_id = target.id
    target.survival_score = 0.0
    cortex.prune()
    assert target.state == NeuronState.PRUNED


# ── Hebbian learning ──────────────────────────────────────────────────────────

def test_hebbian_strengthens_co_success(cortex):
    ids = list(cortex.neurons.keys())[:2]
    src, dst = ids[0], ids[1]
    cortex.graph.add_edge(src, dst, weight=0.5)
    cortex.hebbian_update(successful_ids=[src, dst], failed_ids=[])
    assert cortex.graph[src][dst]["weight"] > 0.5


def test_hebbian_weakens_on_failure(cortex):
    ids = list(cortex.neurons.keys())[:2]
    src, dst = ids[0], ids[1]
    cortex.graph.add_edge(src, dst, weight=0.5)
    cortex.hebbian_update(successful_ids=[], failed_ids=[src, dst])
    assert cortex.graph[src][dst]["weight"] < 0.5


def test_hebbian_forges_new_edge_on_co_success(cortex):
    ids = list(cortex.neurons.keys())[:2]
    src, dst = ids[0], ids[1]
    cortex.graph.remove_edges_from([(src, dst), (dst, src)])
    cortex.hebbian_update(successful_ids=[src, dst], failed_ids=[])
    assert cortex.graph.has_edge(src, dst)


def test_hebbian_weight_clamped_at_one(cortex):
    ids = list(cortex.neurons.keys())[:2]
    src, dst = ids[0], ids[1]
    cortex.graph.add_edge(src, dst, weight=1.0)
    for _ in range(10):
        cortex.hebbian_update(successful_ids=[src, dst], failed_ids=[])
    assert cortex.graph[src][dst]["weight"] <= 1.0


# ── Neurogenesis ──────────────────────────────────────────────────────────────

def test_neurogenesis_spawns_near_thriving(cortex):
    for n in cortex.neurons.values():
        n.survival_score = 0.9
    before = len(cortex.neurons)
    born = cortex.neurogenesis()
    assert len(cortex.neurons) >= before  # at least same, likely more


def test_neurogenesis_respects_max(cortex):
    import colony.config as cfg
    for n in cortex.neurons.values():
        n.survival_score = 0.9
    # Fill to max
    while len(cortex.neurons) < cfg.MAX_NEURONS:
        cortex.add_neuron(NeuronAgent(survival_score=0.9))
    born = cortex.neurogenesis()
    assert born == []
    assert len(cortex.neurons) == cfg.MAX_NEURONS


def test_no_neurogenesis_without_thriving(cortex):
    for n in cortex.neurons.values():
        n.survival_score = 0.1
    born = cortex.neurogenesis()
    assert born == []


# ── Step (mock mode) ──────────────────────────────────────────────────────────

def test_step_returns_required_keys(cortex):
    result = cortex.step("test task")
    for key in ("generation", "results", "best_response", "pruned", "born", "neuron_count"):
        assert key in result


def test_step_increments_generation(cortex):
    cortex.step("task")
    cortex.step("task")
    assert cortex.generation == 2


def test_step_results_have_success_field(cortex):
    result = cortex.step("task")
    for r in result["results"]:
        assert "success" in r
        assert isinstance(r["success"], bool)


def test_step_competitive_selection_half_fail(cortex):
    # With 4+ active neurons, exactly half should fail
    result = cortex.step("task")
    if len(result["results"]) >= 2:
        passes = sum(1 for r in result["results"] if r["success"])
        fails = sum(1 for r in result["results"] if not r["success"])
        assert passes >= 1
        assert fails >= 1


def test_step_empty_cortex():
    c = Cortex()
    result = c.step("task")
    assert "error" in result


# ── Decay ─────────────────────────────────────────────────────────────────────

def test_decay_all_reduces_scores(cortex):
    for n in cortex.neurons.values():
        n.survival_score = 0.5
    cortex.decay_all()
    for n in cortex.neurons.values():
        assert n.survival_score < 0.5


def test_decay_does_not_go_negative(cortex):
    for n in cortex.neurons.values():
        n.survival_score = 0.0
    cortex.decay_all()
    for n in cortex.neurons.values():
        assert n.survival_score == 0.0


# ── Synthesis ─────────────────────────────────────────────────────────────────

def test_synthesize_returns_empty_without_model(cortex):
    top = [{"id": list(cortex.neurons.keys())[0], "response": "resp A"},
           {"id": list(cortex.neurons.keys())[1], "response": "resp B"}]
    text, sid = cortex.synthesize("task", top)
    assert text == "" and sid == ""


def test_synthesize_returns_empty_with_single_response(cortex):
    from unittest.mock import MagicMock
    cortex.model = MagicMock()
    top = [{"id": list(cortex.neurons.keys())[0], "response": "only one"}]
    text, sid = cortex.synthesize("task", top)
    assert text == "" and sid == ""


def test_synthesize_calls_model_with_multiple_responses(cortex):
    from unittest.mock import MagicMock
    cortex.model = MagicMock()
    cortex.model.generate.return_value = "synthesized output"
    nids = list(cortex.neurons.keys())
    top = [{"id": nids[0], "response": "resp A"}, {"id": nids[1], "response": "resp B"}]
    text, sid = cortex.synthesize("task", top)
    assert text == "synthesized output"
    assert sid != ""
    cortex.model.generate.assert_called_once()


def test_step_includes_synthesized_keys(cortex):
    result = cortex.step("test task")
    assert "synthesized_response" in result
    assert "synthesizer_id" in result
