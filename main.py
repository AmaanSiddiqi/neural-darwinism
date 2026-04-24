"""
Colony — Neural Darwinism Agent System
Usage: python main.py [--model] [--generations N] [--task "your task"]
"""
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Must be first — sets HSA env var before torch loads
import colony.config as cfg

console = Console()


def run_demo(use_model: bool = False, generations: int = 20, task: str = "", no_frames: bool = False):
    from colony.graph.cortex import Cortex
    from colony.visualization.renderer import render_cortex, render_history

    model = None
    if use_model:
        from colony.models.model_manager import ModelManager
        model = ModelManager()

    if not task:
        task = "Explain in 2 sentences why neural networks can approximate any function."

    cortex = Cortex(model_manager=model).seed(n=8)
    console.print(Panel(f"[bold cyan]Colony starting — {len(cortex.neurons)} neurons seeded[/bold cyan]"))
    console.print(f"[dim]Task:[/dim] {task}\n")

    history = []
    if not no_frames:
        frames_dir = Path("frames")
        frames_dir.mkdir(exist_ok=True)

    for gen in range(1, generations + 1):
        result = cortex.step(task)
        history.append(result)

        console.print(
            f"[bold]Gen {gen:3d}[/bold] | "
            f"Neurons: [cyan]{result['neuron_count']:2d}[/cyan] | "
            f"Pruned: [red]{len(result['pruned'])}[/red] | "
            f"Born: [green]{len(result['born'])}[/green]"
        )

        if use_model and result.get("results"):
            score_parts = []
            for r in result["results"]:
                color = "green" if r["success"] else "red"
                score_parts.append(f"[cyan]{r['id']}[/cyan]:[{color}]{r['score']:.2f}[/{color}]")
            console.print(f"  Scores: {'  '.join(score_parts)}")
            console.print(f"  [dim]Best ({result['best_score']:.2f}):[/dim] {result['best_response'][:100]}...")

        if not no_frames:
            render_cortex(cortex, output_path=str(frames_dir / f"gen_{gen:04d}.png"),
                          title=f"Generation {gen} | Neurons: {result['neuron_count']}")

        if gen % 5 == 0:
            cortex.status()

    render_cortex(cortex, output_path="cortex_final.png", title="Final State")
    render_history(history, output_path="history.png")
    console.print(Panel("[bold green]Done. Saved cortex_final.png and history.png[/bold green]"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colony — Neural Darwinism")
    parser.add_argument("--model", action="store_true", help="Load real LLM (default: mock mode)")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--task", type=str, default="", help="Task for the colony to solve")
    parser.add_argument("--no-frames", action="store_true", help="Skip per-generation PNG frames")
    args = parser.parse_args()

    run_demo(use_model=args.model, generations=args.generations, task=args.task, no_frames=args.no_frames)
