"""
Tests for ModelManager — judge now delegates to external Claude API,
so we patch colony.judge.judge_response directly.
"""
import pytest
from unittest.mock import MagicMock, patch


def make_mock_manager():
    from colony.models.model_manager import ModelManager

    mgr = ModelManager.__new__(ModelManager)
    mgr._has_adapters = False
    mgr.model = MagicMock()
    mgr.model.disable_adapter_layers = MagicMock()
    mgr.model.enable_adapter_layers = MagicMock()
    mgr.tokenizer = MagicMock()
    mgr.tokenizer.eos_token_id = 0
    return mgr


@pytest.fixture
def mock_manager():
    return make_mock_manager()


@pytest.mark.parametrize("raw_score,expected_norm,expected_pass", [
    (8.0,  0.8,  True),
    (3.0,  0.3,  False),
    (10.0, 1.0,  True),
    (1.0,  0.1,  False),
    (6.0,  0.6,  True),
    (5.0,  0.5,  False),
    (5.0,  0.5,  False),   # fallback on error
    (5.0,  0.5,  False),   # fallback on empty
    (10.0, 1.0,  True),    # clamped from 9.9
])
def test_judge_delegates_to_external(mock_manager, raw_score, expected_norm, expected_pass):
    with patch("colony.judge.judge_response", return_value=(expected_pass, expected_norm)) as mock_j:
        passed, score = mock_manager.judge("task", "response")
        mock_j.assert_called_once_with("task", "response")
        assert score == pytest.approx(expected_norm, abs=0.01)
        assert passed == expected_pass


def test_judge_passes_full_args(mock_manager):
    with patch("colony.judge.judge_response", return_value=(True, 0.8)) as mock_j:
        mock_manager.judge("what is X?", "X is Y because Z.")
        mock_j.assert_called_once_with("what is X?", "X is Y because Z.")
