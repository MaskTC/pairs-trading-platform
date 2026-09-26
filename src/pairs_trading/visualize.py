"""Plot helpers. All figures are saved to disk (Agg backend)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_spread(spread, z, entry_z, exit_z, path, title="Spread and z-score"):
    """Two panels: spread level, then z-score with entry/exit bands."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(spread.index, spread.values, color="steelblue", lw=1.2)
    axes[0].set_title(f"{title} -- spread")
    axes[0].set_ylabel("Spread (A - beta*B)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(z.index, z.values, color="darkorange", lw=1.2, label="z-score")
    axes[1].axhline(entry_z, color="red", ls="--", label=f"+entry {entry_z}")
    axes[1].axhline(-entry_z, color="red", ls="--", label=f"-entry {entry_z}")
    axes[1].axhline(exit_z, color="green", ls=":", label=f"+exit {exit_z}")
    axes[1].axhline(-exit_z, color="green", ls=":", label=f"-exit {exit_z}")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("z-score with entry/exit bands")
    axes[1].set_ylabel("z")
    axes[1].legend(loc="upper left", fontsize="small")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity(equity, benchmark, path, title="Equity curve vs buy-and-hold"):
    """Strategy equity vs a buy-and-hold benchmark, both rebased."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, label="Pairs strategy", lw=1.5)
    ax.plot(
        benchmark.index,
        benchmark.values,
        label="Buy-and-hold benchmark",
        lw=1.2,
        alpha=0.8,
    )
    ax.set_title(title)
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
