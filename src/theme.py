"""Chart & UI theme — Expedia-inspired visual identity, applied to Plotly + the app CSS.

Palette story: Expedia blue (trust, primary brand) + gold (highlights, stars) + green
(verified/positive/improving) + maroon (alert/negative/declining). Color is assigned
by the job it does (see the dataviz method), matching the CSS variables in app.py 1:1:
  * sentiment/polarity in [-1, 1]  -> DIVERGING maroon<->cream<->blue
  * magnitude (counts, evidence)   -> SEQUENTIAL single gold ramp
  * profile / hotel identity        -> CATEGORICAL slots in fixed order (never cycled)
  * status (improving/declining)    -> reserved status palette, never reused for a series
"""
from __future__ import annotations

# ----------------------------------------------------------------- core palette
# these mirror the :root CSS variables in app.py — keep them in sync
INK = "#0F172A"         # near-black navy — headings, primary text
GOLD = "#FFCC00"        # Expedia gold — stars, highlights, badges
GOLD_DARK = "#B45309"   # darker gold for text-on-light (contrast-safe)
BLUE = "#0000FF"        # Expedia blue — primary brand / positive
GREEN = "#059669"       # verified / improving
MAROON = "#E11D48"      # alerts / negative / declining
CREAM = "#F8FAFC"       # page surface
MUTED = "#64748B"       # secondary text
GRID = "#E2E8F0"        # gridlines / borders
SURFACE = "#FFFFFF"     # card background

# categorical slots (fixed order) — brand-safe identity colors.
# Deliberately avoids raw gold as a full-tile background (white text on
# yellow fails contrast); gold stays reserved for small accents/stars.
CATEGORICAL = ["#0000EE", "#0F172A", "#0B6E4F", "#9C2B3C",
               "#6A4C93", "#0E7C86", "#B45309", "#334155"]

# diverging maroon <-> cream <-> blue (for signed sentiment)
DIVERGING = [[0.0, "#9C1C34"], [0.25, "#E0798A"], [0.5, "#F1F5F9"],
             [0.75, "#5B76E0"], [1.0, "#1E3A8A"]]

# sequential gold ramp (for magnitude / evidence volume)
SEQUENTIAL = [[0.0, "#FEF3C7"], [0.5, "#FBBF24"], [1.0, "#B45309"]]

STATUS = {"good": GREEN, "warning": GOLD, "serious": "#EA580C", "critical": MAROON}
IMPROVING, DECLINING = STATUS["good"], STATUS["critical"]


def polarity_color(value: float) -> str:
    """Map a sentiment in [-1, 1] to a diverging color for inline chips."""
    if value >= 0.5:
        return "#047857"
    if value >= 0.15:
        return "#059669"
    if value > -0.15:
        return "#64748B"
    if value > -0.5:
        return "#E0798A"
    return "#E11D48"


def sentiment_icon(value: float) -> str:
    """A small emoji that reinforces the sentiment score at a glance."""
    if value >= 0.5:
        return "😊"
    if value >= 0.15:
        return "🙂"
    if value > -0.15:
        return "😐"
    if value > -0.5:
        return "🙁"
    return "😞"


def star_rating(category: str) -> str:
    """'4-star' -> '★★★★☆' — a visual star strip for hotel category chips."""
    try:
        n = int(str(category).split("-")[0])
    except (ValueError, IndexError):
        n = 3
    n = max(0, min(5, n))
    return "★" * n + "☆" * (5 - n)


def rank_medal(rank: int) -> str:
    """Top-3 recommendation ranks get a medal instead of a plain number."""
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")


_TRAVELER_ICONS = {
    "solo": "🧳", "couple": "💑", "family": "👨‍👩‍👧", "group": "👥", "business": "💼",
}


def traveler_icon(traveler_type: str) -> str:
    return _TRAVELER_ICONS.get(str(traveler_type).lower(), "🧳")


_ARCHETYPE_KEYWORDS = [
    ("luxury", "💎"), ("wellness", "🧘"), ("beach", "🏖️"), ("nightlife", "🎉"),
    ("culture", "🏛️"), ("foodie", "🍽️"), ("business", "💼"), ("budget", "💰"),
    ("family", "👨‍👩‍👧"), ("accessible", "♿"), ("central", "📍"), ("safety", "🛡️"),
]


def archetype_icon(archetype: str) -> str:
    """Pick the most evocative emoji for a traveler archetype string."""
    a = str(archetype).lower()
    for key, icon in _ARCHETYPE_KEYWORDS:
        if key in a:
            return icon
    return "🧭"


def apply_layout(fig, height: int = 340, title: str | None = None):
    fig.update_layout(
        height=height, title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=12, font_family="Plus Jakarta Sans, sans-serif",
                        bgcolor=INK, font_color="#ffffff"),
        colorway=CATEGORICAL,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID, color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=GRID, color=MUTED)
    return fig