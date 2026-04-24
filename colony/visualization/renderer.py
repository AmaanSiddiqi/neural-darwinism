import matplotlib
matplotlib.use("Agg")  # headless-safe; swap to "TkAgg" or "Qt5Agg" for live window
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from pathlib import Path


_CMAP = plt.cm.RdYlGn  # red=dying, yellow=mid, green=thriving


def render_cortex(cortex, output_path: str = "cortex.png", title: str = ""):
    """Render the cortex graph to a PNG. Call after each step for animation frames."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.set_title(title or f"Cortex — Gen {cortex.generation}", color="white", fontsize=14)
    ax.axis("off")

    if not cortex.neurons:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return

    G = cortex.graph
    pos = {nid: (n.x, n.y) for nid, n in cortex.neurons.items()}

    # Edge widths and alpha proportional to weight
    edges = list(G.edges(data=True))
    if edges:
        weights = np.array([d.get("weight", 0.1) for _, _, d in edges])
        edge_colors = [(*mcolors.to_rgb("#4488ff"), float(w)) for w in weights]
        edge_widths = (weights * 4).tolist()
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
        )

    # Node colors from survival score
    node_ids = list(cortex.neurons.keys())
    survival_scores = [cortex.neurons[nid].survival_score for nid in node_ids]
    node_colors = [_CMAP(s) for s in survival_scores]
    node_sizes = [200 + 600 * s for s in survival_scores]

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_ids,
                           node_color=node_colors, node_size=node_sizes, alpha=0.9)

    # Labels: id + survival score
    labels = {nid: f"{nid}\n{cortex.neurons[nid].survival_score:.2f}" for nid in node_ids}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_color="white", font_size=7)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Survival Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)


def render_history(history: list[dict], output_path: str = "history.png"):
    """Plot neuron count, pruning, and neurogenesis over generations."""
    gens = [h["generation"] for h in history]
    counts = [h["neuron_count"] for h in history]
    pruned = [len(h.get("pruned", [])) for h in history]
    born = [len(h.get("born", [])) for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), facecolor="#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("white")

    axes[0].plot(gens, counts, color="#4488ff", linewidth=2, label="Neurons alive")
    axes[0].set_ylabel("Neuron count", color="white")
    axes[0].legend(facecolor="#222222", labelcolor="white")

    axes[1].bar(gens, pruned, color="#ff4444", alpha=0.8, label="Pruned")
    axes[1].bar(gens, born, color="#44ff88", alpha=0.8, label="Born")
    axes[1].set_ylabel("Events", color="white")
    axes[1].set_xlabel("Generation", color="white")
    axes[1].legend(facecolor="#222222", labelcolor="white")

    fig.suptitle("Cortex Lifecycle", color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
