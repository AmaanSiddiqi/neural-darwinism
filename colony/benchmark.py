"""
Holdout benchmark — fixed analytical questions never used as training data.
Run before and after adapter upgrades to measure genuine generalisation.
"""
from colony.judge import judge_response

HOLDOUT_TASKS = [
    "What are the second and third-order consequences of widespread AI adoption on labor markets and social structures?",
    "Analyze the tension between individual privacy and collective security. What frameworks help navigate this trade-off?",
    "What are the strongest arguments against the scientific consensus on a major topic, and how should those arguments be properly evaluated?",
    "How do network effects create winner-take-all dynamics, and under what conditions do they fail to materialise?",
    "What does history tell us about how societies successfully navigate major technological transitions? What patterns emerge?",
]


def run_benchmark(generate_fn) -> dict:
    """
    Score the model on holdout tasks using the external judge.
    generate_fn(prompt: str) -> str
    Returns {"tasks": [{task, score}], "mean_score": float}
    """
    results = []
    for task in HOLDOUT_TASKS:
        response = generate_fn(f"Task: {task}\n\nResponse:")
        _, score = judge_response(task, response)
        results.append({"task": task[:80] + ("..." if len(task) > 80 else ""), "score": round(score, 3)})

    mean = round(sum(r["score"] for r in results) / len(results), 3) if results else 0.0
    return {"tasks": results, "mean_score": mean}
