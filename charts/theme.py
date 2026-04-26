"""
btc_dashboard.charts.theme
Maison theme tokens, frosted-glass CSS injector, and small helpers used by
every page.

Palette story (warm Italian villa interior):
  - oatmeal cream surfaces
  - chocolate brown brand
  - taupe + brass as accents
  - muted sage / terracotta as bull / bear

Type:
  - Italiana          (display · page titles, KPI numbers, card titles)
  - Inter             (body)
  - JetBrains Mono    (numerics, eyebrows, tickers)
"""

from __future__ import annotations

import streamlit as st


# ── Palette (Navy on Clean Beige) ────────────────────────────────────────────
BG          = "#EFE7D6"   # clean beige (page) — single flat tone, no gradient
BG_2        = BG          # kept for compat; identical to BG
SURFACE     = BG
SURFACE_2   = BG

BRAND       = "#1F3A5F"   # deep navy
BRAND_DEEP  = "#14283F"   # darkest navy
BRAND_SOFT  = "#DCE3EE"   # light navy tint
BRAND_LINE  = "#B6C2D6"

ACCENT      = "#4A6FA5"   # steel blue
ACCENT_BRIGHT = "#7A98C4"

BRASS       = "#B69248"   # dusty gold accent (RN mean, runner, flip)
BULL        = "#6B8B68"   # muted forest / olive sage
BEAR        = "#A35A48"   # warm terracotta / clay red

CREAM       = "#FAF6EC"
INK         = "#14283F"   # darkest navy · primary text
INK_DIM     = "#3D5A80"   # mid-navy · secondary text
STONE       = "#7A8AA0"   # muted blue-grey · tertiary / labels
RULE        = "#DDD2BD"

# ── Soft (alpha) versions used in chart fills ────────────────────────────────
BULL_FILL   = "rgba(107,139,104,0.10)"
BEAR_FILL   = "rgba(163,90,72,0.10)"
ACCENT_FILL = "rgba(168,133,96,0.16)"
BRASS_FILL  = "rgba(182,146,72,0.14)"

# ── Chart-specific (warm rule for grids / dividers) ──────────────────────────
GRID        = "#E5DCC9"

# ── Chart line palette · dark blue shades for curves on warm bg ──────────────
NAVY        = "#1F3A5F"   # deep navy — primary chart line (RN density, RN mean)
STEEL       = "#4A6FA5"   # steel blue — secondary chart line (OI-adjusted)

# ── Backwards-compatible aliases (so existing pages keep importing) ──────────
GOLD  = NAVY        # primary chart-line colour
TEAL  = STEEL       # secondary chart-line colour
RED   = BEAR
AMBER = BRASS


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS INJECTION
# Call once per page (after st.set_page_config) to bring fonts + tokens +
# frosted-glass surfaces into the Streamlit shell.
# ─────────────────────────────────────────────────────────────────────────────
def inject_global_css() -> None:
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Italiana&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
  --bg:           #EFE7D6;
  --bg-2:         #EFE7D6;
  --surface:      #EFE7D6;
  --surface-2:    #EFE7D6;

  --brand:        #1F3A5F;
  --brand-deep:   #14283F;
  --brand-soft:   #DCE3EE;
  --brand-line:   #B6C2D6;

  --accent:       #4A6FA5;
  --accent-bright:#7A98C4;

  --brass:        #B69248;
  --brass-soft:   rgba(182,146,72,0.14);
  --bull:         #6B8B68;
  --bull-soft:    rgba(107,139,104,0.14);
  --bear:         #A35A48;
  --bear-soft:    rgba(163,90,72,0.12);

  --ink:          #14283F;
  --ink-dim:      #3D5A80;
  --stone:        #7A8AA0;
  --rule:         #DDD2BD;

  --display: 'Italiana', 'Cormorant Garamond', Georgia, serif;
  --sans:    'Inter', -apple-system, system-ui, sans-serif;
  --mono:    'JetBrains Mono', monospace;

  /* Frosted glass · warm-tinted, translucent, heavily blurred */
  --glass-bg:        linear-gradient(155deg, rgba(255,250,240,0.55) 0%, rgba(255,250,240,0.30) 100%);
  --glass-bg-light:  linear-gradient(155deg, rgba(255,250,240,0.34) 0%, rgba(255,250,240,0.16) 100%);
  --glass-border:    1px solid rgba(255,250,240,0.65);
  --glass-border-soft:1px solid rgba(255,250,240,0.45);
  --glass-shadow:
       0 1px 0 rgba(255,250,240,0.70) inset,
       0 -1px 0 rgba(92,59,37,0.06) inset,
       0 2px 4px rgba(60,40,22,0.06),
       0 18px 48px rgba(60,40,22,0.12);
  --glass-blur:        blur(34px) saturate(170%);
  --glass-blur-soft:   blur(20px) saturate(150%);
}

/* Flat clean beige page · single tone, no gradient, no bokeh */
html, body, [data-testid="stAppViewContainer"], .stApp {
  font-family: var(--sans) !important;
  color: var(--ink) !important;
  font-feature-settings: 'tnum' on, 'lnum' on;
  background: var(--bg) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Streamlit headings → Italiana */
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: var(--display) !important;
  font-weight: 400 !important;
  color: var(--ink) !important;
  letter-spacing: -0.005em;
}

/* Sidebar: solid navy rail · single tone, no gradient */
section[data-testid="stSidebar"] {
  background: var(--brand) !important;
  border-right: 1px solid rgba(20,40,63,0.18);
}
section[data-testid="stSidebar"] * {
  color: rgba(255,250,240,0.85) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
  font-family: var(--display) !important;
  color: #FAF6EC !important;
  font-weight: 400 !important;
}

/* Buttons → frosted-glass ghost; primary stays chocolate */
.stButton > button {
  background: var(--glass-bg-light) !important;
  backdrop-filter: var(--glass-blur-soft);
  -webkit-backdrop-filter: var(--glass-blur-soft);
  border: 1px solid var(--brand-line) !important;
  border-radius: 6px !important;
  color: var(--brand) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  box-shadow: 0 1px 0 rgba(255,250,240,0.65) inset !important;
}
.stButton > button:hover {
  background: rgba(232,220,203,0.55) !important;
  border-color: var(--brand) !important;
  color: var(--brand-deep) !important;
}
section[data-testid="stSidebar"] .stButton > button {
  background: rgba(255,250,240,0.10) !important;
  border-color: rgba(255,250,240,0.25) !important;
  color: #FAF6EC !important;
}

/* Metric tiles → frosted glass · min-height keeps a row aligned even when
   one tile has no delta */
[data-testid="stMetric"] {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: var(--glass-border);
  border-radius: 6px;
  padding: 14px 16px;
  box-shadow: var(--glass-shadow);
  min-height: 110px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  height: 100%;
}
[data-testid="stMetric"] > div { height: 100%; display: flex; flex-direction: column; }
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  letter-spacing: 0.22em !important;
  text-transform: uppercase;
  color: var(--stone) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--display) !important;
  font-weight: 400 !important;
  color: var(--ink) !important;
  font-size: 32px !important;
  line-height: 1 !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

/* Selectbox / inputs */
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  background: var(--glass-bg-light) !important;
  backdrop-filter: var(--glass-blur-soft);
  -webkit-backdrop-filter: var(--glass-blur-soft);
  border: 1px solid var(--brand-line) !important;
  border-radius: 6px !important;
}

/* Plotly chart container → soft frosted card */
[data-testid="stPlotlyChart"], .js-plotly-plot {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: var(--glass-border);
  border-radius: 6px;
  box-shadow: var(--glass-shadow);
  padding: 8px 10px;
}

/* Dataframes / tables */
[data-testid="stDataFrame"] {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: var(--glass-border);
  border-radius: 6px;
  box-shadow: var(--glass-shadow);
}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--ink-dim) !important;
  font-family: var(--sans) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--glass-bg-light);
  backdrop-filter: var(--glass-blur-soft);
  -webkit-backdrop-filter: var(--glass-blur-soft);
  border: var(--glass-border-soft);
  border-radius: 6px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-dim) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--brand) !important;
  color: #FAF6EC !important;
  border-radius: 4px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Small text helpers
# ─────────────────────────────────────────────────────────────────────────────
def section_label(text: str) -> str:
    return (
        f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;'
        f'letter-spacing:.22em;text-transform:uppercase;color:{BRAND};'
        f'margin:22px 0 12px 0;display:flex;align-items:center;gap:10px;">'
        f'<span style="display:inline-block;width:18px;height:1px;'
        f'background:{BRAND};"></span>{text}</div>'
    )


def page_title(title: str, tagline: str) -> None:
    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;'
        f'letter-spacing:0.26em;text-transform:uppercase;color:{BRAND};'
        f'margin-bottom:10px;">{tagline}</div>'
        f'<h1 style="font-family:Italiana,Cormorant Garamond,Georgia,serif;'
        f'font-weight:400;font-size:64px;line-height:1.0;letter-spacing:-0.005em;'
        f'color:{INK};margin:0 0 20px 0;">{title}</h1>',
        unsafe_allow_html=True,
    )


def fmt_money(x) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    if not v == v:  # NaN
        return "—"
    return f"${v:,.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Plotly base layout — Maison-tinted, transparent so the frosted card behind
# the chart shows through.
# ─────────────────────────────────────────────────────────────────────────────
def base_layout(title: str = None, height: int = 320) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color=INK_DIM, size=11),
        height=height,
        margin=dict(l=58, r=20, t=30 if title else 12, b=44),
        title=(
            dict(
                text=title,
                font=dict(family="Italiana, Cormorant Garamond, serif",
                          size=15, color=INK),
                x=0.0, xanchor="left",
            )
            if title else None
        ),
    )
