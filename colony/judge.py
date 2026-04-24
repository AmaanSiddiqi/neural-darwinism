import anthropic
import colony.config as cfg

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    return _client


def judge_response(task: str, response: str) -> tuple[bool, float]:
    """
    Score a response 1-10 using Claude Haiku as external judge.
    Non-circular: the judge model is separate from the colony model.
    Returns (passed, normalised_score).
    """
    prompt = (
        f"Rate the following response to a question on a scale of 1 to 10.\n\n"
        f"Question: {task}\n\n"
        f"Response: {response[:600]}\n\n"
        f"Criteria: analytical depth, logical coherence, coverage of key angles, insight quality.\n"
        f"Reply with ONLY a single integer from 1 to 10. Rating:"
    )
    try:
        msg = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        score = float("".join(c for c in raw if c.isdigit() or c == ".")[:3])
        score = max(1.0, min(10.0, score))
    except Exception as e:
        print(f"[judge] API error: {e} — defaulting to 5.0")
        score = 5.0
    return score >= 6.0, score / 10.0
