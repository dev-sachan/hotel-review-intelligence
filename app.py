"""Hotel Review Intelligence Engine — demo app.

Loads only the cached pipeline artifacts (no model inference), so it starts instantly.
Run:  streamlit run app.py   (after  python -m src.pipeline)
"""
from __future__ import annotations

import json
import base64
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import appdata, config, theme
from src.scoring import profile_conditioned_scores
from src.taxonomy import ASPECT_KEYS, DISPLAY_NAMES

st.set_page_config(page_title="Hotel Review Intelligence", page_icon="🏨", layout="wide")

# ============================================================ IMAGE HELPER
@st.cache_data(show_spinner=False)
def get_base64_image(file_path: str) -> str:
    """Reads a local image and converts it to a base64 string for CSS/HTML injection."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception:
            return ""
    return ""

# Read local images (Ensure logo.png and bg.png are in the same folder as app.py)
logo_b64 = get_base64_image("logo.png")
bg_b64 = get_base64_image("bg.png")

# ============================================================ STATIC STYLING
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

  :root{
    --ink:       #0F172A; 
    --ink-light: #334155; 
    --gold:      #FFCC00; 
    --gold-dark: #B45309;
    --blue:      #0000FF; /* Pure Expedia Blue */
    --blue-dark: #0000CC;
    --maroon:    #E11D48;
    --green:     #059669;
    --cream:     #F8FAFC;
    --muted:     #64748B;
    --grid:      #E2E8F0;
    --surface:   #FFFFFF;
    --border:    #CBD5E1;
  }

  /* Force global typography and color */
  html, body, [class*="css"], .stMarkdown p, .stText, p, span, div {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
  }
  
  h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { 
    font-weight: 800 !important; 
    letter-spacing: -0.03em !important; 
    color: var(--ink) !important;
  }

  /* ================= THE READABILITY FIX (GLASS PANE) ================= */
  .stApp { background-color: transparent !important; }
  
  .block-container {
    background: rgba(255, 255, 255, 0.98) !important;
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 3rem 4rem !important;
    margin-top: 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    max-width: 1300px !important;
  }

  /* ===== TOPBAR ===== */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    border-bottom: 2px solid var(--grid);
  }
  .topbar-brand { display: flex; align-items: center; gap: 16px; }
  .topbar-sub { color: var(--muted); font-size: 0.95rem; font-weight: 600; letter-spacing: 0.02em; border-left: 2px solid var(--grid); padding-left: 16px; }
  .topbar-tag { background: rgba(0,0,255,0.08); border: 1px solid rgba(0,0,255,0.15); color: var(--blue); font-size: 0.75rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; padding: 6px 14px; border-radius: 999px; }

  /* ===== HERO ===== */
  .hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    border-radius: 16px;
    padding: 50px 40px;
    margin-bottom: 32px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
  }
  .hero-badge { display: inline-flex; align-items: center; gap: 8px; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: var(--blue); background: var(--surface); border-radius: 999px; padding: 6px 16px; margin-bottom: 20px; }
  .hero h1 { font-size: 2.6rem !important; font-weight: 800 !important; color: #ffffff !important; margin: 0 0 12px; line-height: 1.15; }
  .hero h1 span { color: var(--gold); }
  .hero p { font-size: 1.1rem !important; color: rgba(255,255,255,0.9) !important; max-width: 820px; margin: 0; line-height: 1.6; font-weight: 500; }
  .hero-stats { display: flex; gap: 0; margin-top: 32px; flex-wrap: wrap; background: rgba(0,0,0,0.3); border-radius: 12px; overflow: hidden; width: fit-content; border: 1px solid rgba(255,255,255,0.15); }
  .hstat { text-align: center; padding: 16px 32px; border-right: 1px solid rgba(255,255,255,0.1); }
  .hstat:last-child { border-right: none; }
  .hnum { display: block; font-size: 1.8rem; font-weight: 800; color: #ffffff; line-height: 1; }
  .hlabel { display: block; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em; color: rgba(255,255,255,0.7); margin-top: 6px; font-weight: 700; }

  /* ===== NAV BAR (elevated segmented control) ===== */
  .stTabs [data-baseweb="tab-list"] { 
    background: var(--surface) !important;
    gap: 4px !important;
    width: 100% !important;
    display: flex !important;
    border: 1px solid var(--grid) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    margin-bottom: 6px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06) !important;
  }
  .stTabs [data-baseweb="tab"] { 
    flex: 1 !important; /* stretch evenly */
    position: relative !important;
    padding: 12px 8px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    display: flex !important;
    justify-content: center !important;
    transition: background 0.15s ease !important;
  }
  .stTabs [data-baseweb="tab"]:hover { background: rgba(0,0,255,0.05) !important; }
  /* Force text colors + weight in tabs */
  .stTabs [data-baseweb="tab"] p { 
    font-weight: 700 !important; 
    font-size: 1.0rem !important;
    color: var(--ink-light) !important;
    margin: 0 !important;
    transition: color 0.15s ease !important;
  }
  /* Reliable active state — driven directly off aria-selected, not Streamlit's
     internal highlight element (which was inconsistently themed and leaking
     the default red instead of brand blue). */
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(0,0,255,0.08) !important;
  }
  .stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: var(--blue) !important;
    font-weight: 800 !important;
  }
  .stTabs [data-baseweb="tab"][aria-selected="true"]::after {
    content: ""; position: absolute; left: 16px; right: 16px; bottom: 2px;
    height: 3px; background: var(--blue); border-radius: 3px;
  }
  /* Neutralize Streamlit's own highlight bar entirely — our ::after above
     is the single source of truth for the active indicator. */
  .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
  .stTabs [data-baseweb="tab-border"] { background-color: transparent !important; }

  /* ===== SECTION HEADERS ===== */
  .sec-head { display: flex; align-items: center; gap: 16px; margin: 24px 0; }
  .sec-icon { width: 48px; height: 48px; border-radius: 12px; flex-shrink: 0; background: rgba(0,0,255,0.08); color: var(--blue); display: flex; align-items: center; justify-content: center; font-size: 1.3rem; }
  .sec-title { font-size: 1.6rem; font-weight: 800; color: var(--ink); line-height: 1.2; position: relative; display: inline-block; z-index: 1; margin:0; }
  .sec-title::after { content: ''; position: absolute; bottom: 2px; left: -4px; width: calc(100% + 8px); height: 35%; background: rgba(255, 204, 0, 0.4); z-index: -1; transform: skew(-10deg); border-radius: 2px; }
  .sec-sub { font-size: 0.95rem; color: var(--ink-light); margin-top: 4px; font-weight: 500; }

  /* ===== CARDS ===== */
  .rec-card, .stat-card {
    background: var(--surface); 
    border: 1px solid var(--grid);
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: all 0.2s;
  }
  .rec-card { position: relative; padding: 22px; margin-bottom: 16px; border-top: 4px solid var(--blue); }
  .rec-card:hover { transform: translateY(-3px); box-shadow: 0 12px 20px -8px rgba(0, 0, 255, 0.15); border-color: var(--blue); }
  .rec-card.contra-card { border-top-color: var(--maroon); }
  
  .medal { position: absolute; top: -16px; right: 20px; font-size: 1.8rem; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1)); }
  .rec-rank { font-size: 0.75rem; font-weight: 800; color: var(--blue); letter-spacing: 0.08em; text-transform: uppercase; }
  .rec-hotel { font-size: 1.25rem; font-weight: 800; color: var(--ink); margin: 6px 0; }
  .rec-meta { font-size: 0.85rem; color: var(--muted); margin-bottom: 12px; display: flex; align-items: center; gap: 10px; font-weight: 600; }
  .stars { color: var(--gold-dark); letter-spacing: 2px; font-size: 0.95rem; }
  .chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; margin: 4px 6px 4px 0; border: 1px solid var(--grid); background: var(--cream); color: var(--ink) !important; }
  .quote { position: relative; font-size: 0.95rem; color: var(--ink); border-left: 3px solid var(--blue); padding: 12px 16px; margin: 12px 0; background: rgba(0,0,255,0.04); border-radius: 0 8px 8px 0; font-style: italic; line-height: 1.6; }
  .quote-meta { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 0.75rem; font-style: normal; margin-top: 8px; font-weight: 700; }

  /* ===== CAVEAT / WARNING NOTE ===== */
  .caveat {
    display: flex; align-items: flex-start; gap: 10px;
    font-size: 0.82rem; color: #8a5200; font-weight: 600;
    background: linear-gradient(90deg, #FFFBEB, #FFF3D6);
    border: 1px solid #FDE68A; border-left: 4px solid var(--gold);
    border-radius: 8px; padding: 10px 14px; margin-top: 12px; line-height: 1.5;
  }
  .caveat i { color: var(--gold-dark); margin-top: 2px; }

  /* ===== TREND ROWS (Portfolio pulse movers) ===== */
  .trend-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 11px 15px; border-radius: 10px; margin-bottom: 8px;
    background: var(--surface); border: 1px solid var(--grid);
    font-size: 0.88rem; font-weight: 600; color: var(--ink);
    transition: background 0.15s, transform 0.15s;
  }
  .trend-row:hover { background: var(--cream); transform: translateX(2px); }
  .trend-arrow { margin-right: 8px; font-size: 0.95rem; }

  /* ===== PROFILE HEADER ===== */
  .prof-head { border-radius: 12px; padding: 20px; color: #ffffff; margin-bottom: 16px; position: relative; overflow: hidden; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
  .prof-head::after { content: ""; position: absolute; inset: 0; background: radial-gradient(circle at 100% 0%, rgba(255,255,255,0.2), transparent 60%); }
  .prof-avatar { width: 44px; height: 44px; border-radius: 50%; background: rgba(255,255,255,0.25); display: inline-flex; align-items: center; justify-content: center; font-size: 1.25rem; margin-right: 12px; vertical-align: middle; }
  .prof-name  { font-weight: 800; font-size: 1.1rem; vertical-align: middle; color: #ffffff !important; }
  .prof-desc  { display: block; font-size: 0.85rem; opacity: 0.95; margin-top: 8px; position: relative; font-weight: 500; line-height: 1.4; color: #ffffff !important;}

  /* ===== STAT CARDS ===== */
  .stat-card { padding: 24px 16px; text-align: center; }
  .stat-icon { font-size: 1.8rem; margin-bottom: 8px; }
  .stat-num { font-size: 2.2rem; font-weight: 800; color: var(--ink) !important; margin: 0; line-height: 1.1; }
  .stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted) !important; margin-top: 6px; font-weight: 800; }

  /* ===== DATAFRAMES & MISC ===== */
  [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
  div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius: 8px !important; border-color: var(--border) !important; background: var(--surface) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================ DYNAMIC BACKGROUND
if bg_b64:
    st.markdown(f"""
    <style>
    .stApp {{
        background: url('data:image/png;base64,{bg_b64}') no-repeat center center fixed !important;
        background-size: cover !important;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp { background: #E2E8F0 !important; }</style>", unsafe_allow_html=True)

# ============================================================ HELPER FUNCTIONS
def section_header(fa_icon_class: str, title: str, subtitle: str | None = None):
    sub = f"<div class='sec-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='sec-head'>"
        f"<div class='sec-icon'><i class='{fa_icon_class}'></i></div>"
        f"<div><div class='sec-title'>{title}</div>{sub}</div></div>",
        unsafe_allow_html=True)

def stat_card(fa_icon_class: str, value, label: str, color: str | None = None) -> str:
    style = f" style='color:{color} !important;'" if color else ""
    icon_style = f" style='color:{color} !important;'" if color else " style='color:var(--blue) !important;'"
    return (f"<div class='stat-card'>"
            f"<div class='stat-icon'{icon_style}><i class='{fa_icon_class}'></i></div>"
            f"<div class='stat-num'{style}>{value}</div>"
            f"<div class='stat-label'>{label}</div></div>")

@st.cache_data(show_spinner=False)
def _load():
    return appdata.load_all()

@st.cache_data(show_spinner=False)
def _rich(pid):
    return appdata.load_rich_output(pid)

if not appdata.artifacts_exist():
    st.error("Cached artifacts not found. Run the pipeline first:\n\n"
             "```\npython -m src.pipeline\n```")
    st.stop()

D = _load()
hotels = D["hotels"]
instances = D["instances"]
profiles_by_id = D["profiles_by_id"]
hname = dict(zip(hotels["hotel_id"], hotels["hotel_name"]))

# ============================================================ TOPBAR (BULLETPROOF LOGO)
# If local logo.png exists, use it. Otherwise, generate a clean CSS text logo that matches Expedia branding.
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Expedia Group" style="height: 28px;">'
else:
    logo_html = '<span style="font-weight: 800; font-size: 1.4rem; color: #0000FF; letter-spacing: -1px;"><i class="fa-solid fa-plane-departure" style="margin-right: 6px;"></i> expedia group</span>'

st.markdown(f"""
<div class="topbar">
  <div class="topbar-brand">
    {logo_html}
    <span class="topbar-sub">Hotel Review Intelligence Engine</span>
  </div>
  <span class="topbar-tag">✦ Hackathon 2026 ✦</span>
</div>
""", unsafe_allow_html=True)

# ============================================================ HERO
st.markdown(f"""
<div class="hero">
  <div class="hero-badge"><i class="fa-solid fa-concierge-bell"></i> AI-Powered Concierge Intelligence</div>
  <h1>Find the right hotel<br><span>for every traveler.</span></h1>
  <p>Evidence-based, personalized recommendations distilled from
  <b>{hotels['n_reviews'].sum():,}</b> guest reviews across <b>{len(hotels)}</b> hotels —
  with temporal drift detection, aspect-level sentiment, and contradiction handling
  across reviewer segments.</p>
  <div class="hero-stats">
    <div class="hstat"><span class="hnum">{len(hotels)}</span><span class="hlabel">Hotels tracked</span></div>
    <div class="hstat"><span class="hnum">{hotels['n_reviews'].sum()//1000}K</span><span class="hlabel">Guest reviews</span></div>
    <div class="hstat"><span class="hnum">{len(ASPECT_KEYS)}</span><span class="hlabel">Aspects scored</span></div>
    <div class="hstat"><span class="hnum">{len(profiles_by_id)}</span><span class="hlabel">Traveler profiles</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_rec, tab_hotel, tab_pulse, tab_search = st.tabs(
    ["🎯  Recommendations", "🔍  Hotel deep-dive", "📊  Portfolio pulse", "🔎  Evidence search"])


# ============================================================ TAB 1: RECOMMENDATIONS
def profile_label(pid):
    p = profiles_by_id[pid]
    return f"{pid} · {p['archetype']}"

with tab_rec:
    section_header("fa-solid fa-bullseye", "Compare recommendations across travelers",
                   "Pick up to four profiles to see how the same 120 hotels rank differently "
                   "for different people — every pick is backed by real review quotes.")

    all_ids = sorted(profiles_by_id.keys())
    default = [p for p in ["P01", "P29", "P30", "P08"] if p in all_ids]
    chosen = st.multiselect("Traveler profiles", all_ids, default=default,
                            format_func=profile_label, max_selections=4)

    if not chosen:
        st.info("Select at least one profile above.")
    else:
        cols = st.columns(len(chosen))
        for col, pid in zip(cols, chosen):
            p = profiles_by_id[pid]
            rich = _rich(pid)
            accent = theme.CATEGORICAL[all_ids.index(pid) % len(theme.CATEGORICAL)]
            icon = theme.archetype_icon(p["archetype"])
            with col:
                st.markdown(
                    f"<div class='prof-head' style='background:linear-gradient(135deg,{accent} 0%,{accent}dd 100%)'>"
                    f"<span class='prof-avatar'>{icon}</span>"
                    f"<span class='prof-name'>{pid} — {p['archetype'].replace('_', ' ').title()}</span>"
                    f"<span class='prof-desc'>{p['description']}</span></div>",
                    unsafe_allow_html=True)
                dims = " ".join(
                    f"<span class='chip' style='border-color:rgba(0,0,255,0.2);'>"
                    f"<i class='fa-solid fa-crosshairs' style='color:var(--blue)'></i> {DISPLAY_NAMES.get(d, d)}</span>" for d in p["desired_dims"])
                st.markdown(f"<div style='margin-bottom:16px'>{dims}</div>", unsafe_allow_html=True)

                if not rich:
                    st.warning("No output for this profile.")
                    continue
                for r in rich["recommendations"]:
                    chips = " ".join(
                        f"<span class='chip' style='border-color:rgba(5,150,105,0.2);'>"
                        f"<i class='fa-solid fa-check' style='color:var(--green)'></i> {DISPLAY_NAMES.get(a, a)}</span>" for a in r["matched_aspects"])
                    medal = theme.rank_medal(r["rank"])
                    card = [
                        "<div class='rec-card'>",
                        f"<div class='medal'>{medal}</div>" if medal else "",
                        f"<div class='rec-rank'>Rank #{r['rank']} &nbsp;·&nbsp; Score {r['score']}</div>",
                        f"<div class='rec-hotel'>{r['hotel_name']}</div>",
                        f"<div class='rec-meta'><span class='stars'>{theme.star_rating(r['hotel_category'])}</span>"
                        f"<span>{r['hotel_category']}</span></div>",
                        f"<div>{chips}</div>",
                    ]
                    for e in r["supporting_evidence"][:2]:
                        badge_cls = "color:var(--green)" if e["verified"] else "color:var(--muted)"
                        badge_icon = "<i class='fa-solid fa-circle-check'></i>" if e["verified"] else "<i class='fa-regular fa-circle'></i>"
                        badge_txt = "verified" if e["verified"] else "unverified"
                        card.append(f"<div class='quote'>&ldquo;{e['quote']}&rdquo;"
                                    f"<span class='quote-meta'><i class='fa-solid fa-hotel'></i> {e['review_id']} · {e['date']} · "
                                    f"<span style='{badge_cls}'>{badge_icon} {badge_txt}</span></span></div>")
                    if r["caveats"]:
                        card.append(f"<div class='caveat'><i class='fa-solid fa-triangle-exclamation' style='margin-top:2px'></i> <span>{r['caveats']}</span></div>")
                    card.append("</div>")
                    st.markdown("".join(card), unsafe_allow_html=True)


# ============================================================ TAB 2: HOTEL DEEP-DIVE
with tab_hotel:
    section_header("fa-solid fa-magnifying-glass-chart", "Hotel performance & aspect profile",
                   "Rating trends, seasonal patterns, and where reviewers disagree — one hotel at a time.")
    hsel = st.selectbox("Hotel", hotels["hotel_id"].tolist(),
                        format_func=lambda h: f"{h} · {hname[h]}", key="hotel_pick")
    hrow = hotels[hotels["hotel_id"] == hsel].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("fa-solid fa-pen-to-square", f"{int(hrow['n_reviews']):,}", "Reviews", color="var(--ink)"), unsafe_allow_html=True)
    c2.markdown(stat_card("fa-solid fa-star", f"{hrow['mean_rating']:.2f}", "Mean rating", color="var(--gold-dark)"), unsafe_allow_html=True)
    c3.markdown(stat_card("fa-solid fa-tag",
                          f"<span class='stars' style='font-size:1.3rem'>{theme.star_rating(hrow['hotel_category'])}</span>",
                          hrow["hotel_category"], color="var(--muted)"), unsafe_allow_html=True)
    c4.markdown(stat_card("fa-solid fa-circle-check", f"{hrow['verified_share'] * 100:.0f}%", "Verified share", color="var(--green)"), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("**<i class='fa-solid fa-chart-line' style='color:var(--blue);margin-right:6px;'></i> Overall rating over time**", unsafe_allow_html=True)
        m = D["monthly"]
        mo = m[(m["hotel_id"] == hsel) & (m["aspect"] == "overall")].sort_values("date")
        if len(mo):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mo["date"], y=mo["value"], mode="lines+markers",
                line=dict(color="#0000FF", width=3),
                marker=dict(size=8, color="#FFCC00", line=dict(width=2, color="#0F172A")),
                fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.05)",
                name="mean rating", hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
            dov = D["drift"][(D["drift"]["hotel_id"] == hsel) & (D["drift"]["aspect"] == "overall")]
            if len(dov) and dov.iloc[0]["trend_tier"] not in ("none", "negligible"):
                dd = dov.iloc[0]
                col = theme.IMPROVING if dd["trend_dir"] == "improving" else theme.DECLINING
                arrow = "▲" if dd["trend_dir"] == "improving" else "▼"
                fig.add_annotation(
                    x=mo["date"].iloc[-1], y=mo["value"].iloc[-1],
                    text=f"{arrow} {dd['trend_dir']} ({dd['trend_tier']})",
                    showarrow=True, arrowhead=2,
                    font=dict(color=col, size=12, family="Plus Jakarta Sans"),
                    arrowcolor=col, bgcolor="#fff", bordercolor=col, borderwidth=1, borderpad=6)
            theme.apply_layout(fig, height=300)
            fig.update_yaxes(range=[1, 5])
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**<i class='fa-solid fa-compass' style='color:var(--blue);margin-right:6px;'></i> Aspect scorecard** vs portfolio avg", unsafe_allow_html=True)
        sc = D["scores"]
        hs = sc[sc["hotel_id"] == hsel].set_index("aspect")
        priors = D["scores"].attrs.get("priors", {})
        rows = []
        for a in ASPECT_KEYS:
            if a in hs.index:
                rows.append((DISPLAY_NAMES[a], hs.loc[a, "score"], priors.get(a, 0.0)))
        rows.sort(key=lambda x: x[1])
        adf = pd.DataFrame(rows, columns=["aspect", "score", "prior"])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=adf["aspect"], x=adf["score"], orientation="h",
            marker=dict(color=adf["score"], colorscale=theme.DIVERGING, cmin=-1, cmax=1, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            y=adf["aspect"], x=adf["prior"], mode="markers",
            marker=dict(symbol="line-ns", size=14, color="#0F172A", line=dict(width=2, color="#0F172A")),
            name="portfolio avg", hoverinfo="skip"))
        theme.apply_layout(fig, height=300)
        fig.update_xaxes(range=[-1, 1], showgrid=True, gridcolor="#E2E8F0")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- THIS WAS THE MISSING RESTORED BLOCK ---
    left2, right2 = st.columns([3, 2])
    with left2:
        st.markdown("**<i class='fa-regular fa-calendar-days' style='color:var(--blue);margin-right:6px;'></i> Seasonal sentiment by aspect**", unsafe_allow_html=True)
        asp = instances[(instances["hotel_id"] == hsel) & (~instances["is_general"])]
        piv = asp.pivot_table(index="aspect", columns="season", values="polarity", aggfunc="mean")
        piv = piv.reindex(columns=[s for s in config.SEASON_ORDER if s in piv.columns])
        if len(piv):
            piv.index = [DISPLAY_NAMES.get(a, a) for a in piv.index]
            fig = go.Figure(go.Heatmap(
                z=piv.values, x=[s.capitalize() for s in piv.columns], y=piv.index,
                colorscale=theme.DIVERGING, zmid=0, zmin=-1, zmax=1,
                hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>",
                colorbar=dict(title="sentiment", thickness=12, tickfont=dict(size=11))))
            theme.apply_layout(fig, height=360)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
            
    with right2:
        st.markdown("**<i class='fa-solid fa-scale-balanced' style='color:var(--blue);margin-right:6px;'></i> Where reviewers disagree**", unsafe_allow_html=True)
        contra = D["contradictions"]
        hc = contra[contra["hotel_id"] == hsel].sort_values("disagreement", ascending=False)
        if len(hc) == 0:
            st.caption("No strong contradictions detected for this hotel. <i class='fa-solid fa-check' style='color:var(--green)'></i>", unsafe_allow_html=True)
        for _, c in hc.head(5).iterrows():
            res = c["resolution"] if isinstance(c["resolution"], str) else "Mixed views, no clear driver."
            st.markdown(
                f"<div class='rec-card contra-card' style='margin-bottom:12px; padding: 16px'>"
                f"<b>{DISPLAY_NAMES.get(c['aspect'], c['aspect'])}</b> "
                f"<span class='chip' style='background:#FFF1F2;color:var(--maroon);border-color:rgba(225,29,72,0.2)'>"
                f"<i class='fa-solid fa-bolt'></i> disagreement {c['disagreement']:.2f}</span><br>"
                f"<span style='font-size:0.85rem;color:var(--muted);display:inline-block;margin-top:6px;font-weight:600;'>{res}</span></div>",
                unsafe_allow_html=True)


# ============================================================ TAB 3: PORTFOLIO PULSE
with tab_pulse:
    section_header("fa-solid fa-chart-pie", "Portfolio pulse — the whole estate at a glance",
                   "Leaderboard, fastest movers, and FDR-corrected seasonal alerts across all 120 hotels.")
    drift = D["drift"]

    improving = drift[(drift["aspect"] == "overall") & (drift["trend_dir"] == "improving") & (drift["trend_tier"].isin(["strong", "moderate"]))]
    declining = drift[(drift["aspect"] == "overall") & (drift["trend_dir"] == "declining") & (drift["trend_tier"].isin(["strong", "moderate"]))]
    seasonal = drift[(drift["aspect"] != "overall") & (drift["seasonal_tier"].isin(["strong", "moderate"]))]

    m1, m2, m3 = st.columns(3)
    m1.markdown(stat_card("fa-solid fa-arrow-trend-up", len(improving), "Hotels improving", theme.IMPROVING), unsafe_allow_html=True)
    m2.markdown(stat_card("fa-solid fa-arrow-trend-down", len(declining), "Hotels declining", theme.DECLINING), unsafe_allow_html=True)
    m3.markdown(stat_card("fa-solid fa-cloud-sun-rain", len(seasonal), "Seasonal aspect signals", "var(--gold-dark)"), unsafe_allow_html=True)

    st.write("")
    st.markdown("**<i class='fa-solid fa-trophy' style='color:var(--blue);margin-right:6px;'></i> Hotel leaderboard** (shrunk overall rating)", unsafe_allow_html=True)
    lb = hotels.sort_values("shrunk_rating", ascending=False).reset_index(drop=True)
    lb.insert(0, "Rank", [theme.rank_medal(i + 1) or f"#{i + 1}" for i in range(len(lb))])
    lb_disp = lb[["Rank", "hotel_name", "hotel_category", "n_reviews", "mean_rating", "shrunk_rating"]]
    st.dataframe(
        lb_disp, use_container_width=True, height=290, hide_index=True,
        column_config={
            "Rank":          st.column_config.TextColumn("Rank", width="small"),
            "hotel_name":    st.column_config.TextColumn("Hotel"),
            "hotel_category":st.column_config.TextColumn("Category"),
            "n_reviews":     st.column_config.NumberColumn("Reviews"),
            "mean_rating":   st.column_config.NumberColumn("Avg rating", format="%.2f ⭐"),
            "shrunk_rating": st.column_config.ProgressColumn("Adjusted rating", min_value=1, max_value=5, format="%.2f"),
        })

    st.write("")
    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"**Improving fastest** <i class='fa-solid fa-caret-up' style='color:{theme.IMPROVING}'></i>", unsafe_allow_html=True)
        top_imp = (improving.merge(hotels[["hotel_id", "hotel_name"]], on="hotel_id").sort_values("trend_rho", ascending=False).head(8))
        if len(top_imp) == 0:
            st.caption("No strong improvers detected.")
        for _, r in top_imp.iterrows():
            st.markdown(
                f"<div class='trend-row'><span>"
                f"<i class='fa-solid fa-caret-up trend-arrow' style='color:{theme.IMPROVING}'></i>"
                f"{r['hotel_name']}</span>"
                f"<span class='chip' style='background:#ECFDF5;color:{theme.IMPROVING} !important;border-color:rgba(5,150,105,0.2)'>"
                f"ρ={r['trend_rho']:+.2f} · {r['trend_tier']}</span></div>", unsafe_allow_html=True)
    with cB:
        st.markdown(f"**Declining fastest** <i class='fa-solid fa-caret-down' style='color:{theme.DECLINING}'></i>", unsafe_allow_html=True)
        top_dec = (declining.merge(hotels[["hotel_id", "hotel_name"]], on="hotel_id").sort_values("trend_rho").head(8))
        if len(top_dec) == 0:
            st.caption("No strong decliners detected.")
        for _, r in top_dec.iterrows():
            st.markdown(
                f"<div class='trend-row'><span>"
                f"<i class='fa-solid fa-caret-down trend-arrow' style='color:{theme.DECLINING}'></i>"
                f"{r['hotel_name']}</span>"
                f"<span class='chip' style='background:#FFF1F2;color:{theme.DECLINING} !important;border-color:rgba(225,29,72,0.2)'>"
                f"ρ={r['trend_rho']:+.2f} · {r['trend_tier']}</span></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("**<i class='fa-solid fa-cloud-sun-rain' style='color:var(--blue);margin-right:6px;'></i> Seasonal alerts** — aspects that swing by season (FDR-corrected)", unsafe_allow_html=True)
    sa = (seasonal.merge(hotels[["hotel_id", "hotel_name"]], on="hotel_id").sort_values("season_gap", ascending=False).head(12))
    if len(sa) == 0:
        st.caption("No hotel×aspect combination currently clears the FDR-corrected confidence bar for a seasonal signal. "
                   "That's an honest result of correcting for ~1,070 simultaneous tests, not an empty state bug.")
    else:
        sa_disp = sa[["hotel_name", "aspect", "best_season", "worst_season", "season_gap", "seasonal_tier"]].copy()
        sa_disp["aspect"] = sa_disp["aspect"].map(lambda a: DISPLAY_NAMES.get(a, a))
        st.dataframe(sa_disp, use_container_width=True, hide_index=True,
                     column_config={
                         "hotel_name":   st.column_config.TextColumn("Hotel"),
                         "aspect":       st.column_config.TextColumn("Aspect"),
                         "best_season":  st.column_config.TextColumn("Best season"),
                         "worst_season": st.column_config.TextColumn("Worst season"),
                         "season_gap":   st.column_config.NumberColumn("Gap", format="%.2f"),
                         "seasonal_tier":st.column_config.TextColumn("Confidence"),
                     })

    st.write("")
    exp_seasonal = (drift[(drift["aspect"] != "overall") & (drift["kw_p"] < 0.05)]
                    .merge(hotels[["hotel_id", "hotel_name"]], on="hotel_id")
                    .sort_values("kw_p").head(10))
    exp_trend = (drift[(drift["aspect"] == "overall") & (drift["trend_p"] < 0.05)]
                 .merge(hotels[["hotel_id", "hotel_name"]], on="hotel_id")
                 .sort_values("trend_p").head(10))

    ce1, ce2 = st.columns(2)
    with ce1:
        st.markdown("*Seasonal — raw p < 0.05 (uncorrected)*")
        if len(exp_seasonal) == 0:
            st.caption("None even at the uncorrected level.")
        else:
            ed = exp_seasonal[["hotel_name", "aspect", "best_season", "worst_season", "season_gap", "kw_p"]].copy()
            ed["aspect"] = ed["aspect"].map(lambda a: DISPLAY_NAMES.get(a, a))
            st.dataframe(ed, use_container_width=True, hide_index=True,
                        column_config={
                            "hotel_name": st.column_config.TextColumn("Hotel"),
                            "aspect": st.column_config.TextColumn("Aspect"),
                            "best_season": st.column_config.TextColumn("Best"),
                            "worst_season": st.column_config.TextColumn("Worst"),
                            "season_gap": st.column_config.NumberColumn("Gap", format="%.2f"),
                            "kw_p": st.column_config.NumberColumn("Raw p", format="%.3f"),
                        })
    with ce2:
        st.markdown("*Trend — raw p < 0.05 (uncorrected)*")
        if len(exp_trend) == 0:
            st.caption("None even at the uncorrected level.")
        else:
            ed2 = exp_trend[["hotel_name", "trend_dir", "trend_rho", "trend_p"]].copy()
            st.dataframe(ed2, use_container_width=True, hide_index=True,
                        column_config={
                            "hotel_name": st.column_config.TextColumn("Hotel"),
                            "trend_dir": st.column_config.TextColumn("Direction"),
                            "trend_rho": st.column_config.NumberColumn("ρ", format="%.3f"),
                            "trend_p": st.column_config.NumberColumn("Raw p", format="%.3f"),
                        })

# ============================================================ TAB 4: EVIDENCE SEARCH
with tab_search:
    section_header("fa-solid fa-magnifying-glass", "Semantic evidence search",
                   "Free-text search over the review corpus using sentence embeddings — "
                   "the retrieval layer that supplies recommendation evidence.")

    st.write("") # small spacing
    q = st.text_input("Search reviews", value="quiet room with great wifi for working",
                      key="search_q", label_visibility="collapsed",
                      placeholder="Search reviews — e.g. 'quiet room with great wifi for working'")
    c1, c2 = st.columns(2)
    hfilter = c1.selectbox("Filter by hotel", ["(any)"] + hotels["hotel_id"].tolist(),
                           format_func=lambda h: h if h == "(any)" else f"{h} · {hname[h]}")
    afilter = c2.selectbox("Filter by aspect", ["(any)"] + ASPECT_KEYS,
                           format_func=lambda a: a if a == "(any)" else DISPLAY_NAMES.get(a, a))
    
    if st.button("Search database", type="primary", use_container_width=True):
        pass # Retrieval logic handled below
    st.write("") # small spacing

    if q:
        from src import retrieval
        try:
            res = retrieval.search(q, instances, k=12,
                                   hotel_id=None if hfilter == "(any)" else hfilter,
                                   aspect=None if afilter == "(any)" else afilter)
        except Exception:
            st.warning("Semantic search is warming up its embedding model — this only "
                       "happens once per session (a small ~80MB download on first use). "
                       "Please try the search again in a few seconds.")
            res = pd.DataFrame()
        if len(res) == 0:
            st.info("No matching evidence.")
        for _, r in res.iterrows():
            col = theme.polarity_color(r["polarity"])
            icon = theme.sentiment_icon(r["polarity"])
            badge_cls = "color:var(--green)" if r["verified"] else "color:var(--muted)"
            badge_icon = "<i class='fa-solid fa-circle-check'></i>" if r["verified"] else "<i class='fa-regular fa-circle'></i>"
            badge_txt = "verified" if r["verified"] else "unverified"
            st.markdown(
                f"<div class='rec-card'>"
                f"<div class='rec-meta' style='margin-bottom:6px'>"
                f"<span class='chip' style='border-color:rgba(0,0,255,0.2);'><i class='fa-solid fa-tag' style='color:var(--blue)'></i> {DISPLAY_NAMES.get(r['aspect'], r['aspect'])}</span>"
                f"<span class='chip' style='border-color:rgba(0,0,0,0.1); color:{col} !important;'>{icon} sentiment {r['polarity']:+.2f}</span>"
                f"<span style='margin-left:auto;color:var(--muted);font-size:0.75rem;font-weight:700'><i class='fa-solid fa-bullseye'></i> match {r['similarity']:.2f}</span></div>"
                f"<div class='quote'>&ldquo;{r['sentence_text']}&rdquo;"
                f"<span class='quote-meta'><i class='fa-solid fa-hotel'></i> {r['hotel_name']} · {r['review_id']} · {str(pd.to_datetime(r['review_date']).date())} · <span style='{badge_cls}'>{badge_icon} {badge_txt}</span></span></div>"
                f"</div>",
                unsafe_allow_html=True)