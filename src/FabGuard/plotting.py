"""Shared figure style so EDA, model and presentation figures look consistent."""

import matplotlib as mpl

from FabGuard import config

# Categorical slots in fixed order (colorblind-validated as a set)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

CLASS_COLORS = {0: BLUE, 1: ORANGE}
CLASS_NAMES = {0: "pass", 1: "fail"}
MODEL_COLORS = {"logreg": BLUE, "isolation_forest": ORANGE, "xgboost": AQUA, "dummy": MUTED}
MODEL_NAMES = {
    "dummy": "Dummy",
    "logreg": "Logistic Regression",
    "isolation_forest": "Isolation Forest",
    "xgboost": "XGBoost",
}


def set_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def savefig(fig, name: str):
    """Save ``fig`` to Reports/Figures/<name>.png."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
