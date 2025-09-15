# app.py
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_js_eval import streamlit_js_eval
import re
import plotly.io as pio

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

# ---------------------------
# Config / constants
# ---------------------------
DATAPATH = Path("./data")
ONE_HOUR_MS = 60 * 60 * 1000

st.set_page_config(layout="wide")
st.title("Listing sweet spot explorer")

# ---------------------------
# Utilities (cached IO)
# ---------------------------
@st.cache_data(show_spinner=False)
def get_data_list() -> list[str]:
    """Return list of 'exchange-base' (without .csv.gz) from ./data."""
    return [p.stem.replace(".csv", "") for p in DATAPATH.glob("*.csv.gz")]

@st.cache_data(show_spinner=False)
def get_listings() -> dict:
    with open(DATAPATH / "listings.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_matrix(path: Path) -> pd.DataFrame:
    """
    Load wide matrix CSV.GZ:
      - index: timestamp_a (datetime)
      - columns: timestamp_b (datetime)
      - values: ret_pct
    """
    df = pd.read_csv(path, compression="gzip", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    return df

def nearest_index_val(ts_index, t: pd.Timestamp):
    """Return the nearest timestamp in an Index to t (or None if empty)."""
    if t is None or len(ts_index) == 0:
        return None
    arr = pd.to_datetime(ts_index).values.astype("datetime64[ns]")
    i = int(np.argmin(np.abs(arr - np.datetime64(t))))
    return ts_index[i]

def clamp_window(win_from: pd.Timestamp, win_to: pd.Timestamp,
                 data_from: pd.Timestamp, data_to: pd.Timestamp):
    """Clamp (from,to) window to data bounds."""
    wf = max(win_from, data_from)
    wt = min(win_to,   data_to)
    return (wf, wt)

def add_ref_line(fig, *, orientation: str, pos_dt, mat_win: pd.DataFrame,
                 name: str, color: str, legendgroup: str, showlegend: bool):
    """
    Add a reference line as a Scatter trace so it participates in the legend.
    orientation: 'v' or 'h'
    pos_dt: datetime for the line
    """
    if pos_dt is None:
        return
    pos = pd.Timestamp(pos_dt).to_pydatetime()

    if orientation == "v":
        # skip if out of x range
        if pos < mat_win.columns.min() or pos > mat_win.columns.max():
            return
        fig.add_trace(go.Scatter(
            x=[pos, pos],
            y=[mat_win.index.min(), mat_win.index.max()],
            mode="lines",
            line=dict(color=color, dash="dash", width=2),
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo="skip"
        ))
    else:
        # skip if out of y range
        if pos < mat_win.index.min() or pos > mat_win.index.max():
            return
        fig.add_trace(go.Scatter(
            x=[mat_win.columns.min(), mat_win.columns.max()],
            y=[pos, pos],
            mode="lines",
            line=dict(color=color, dash="dash", width=2),
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo="skip"
        ))

# ---------------------------
# Load data & selections
# ---------------------------
listings = get_listings()
listings_map = {f'{l["exchange"]}-{l["base"]}': l for l in listings["data"]}

entries = []
for k in listings_map.keys():
    if not (DATAPATH / f"{k}.csv.gz").exists():
        continue
    entries.append(k+f' ({pd.to_datetime(listings_map[k]["trade_at"], unit="ms")})')

raw_pair = st.selectbox("Pair", entries)
pair = raw_pair.split(" (")[0]
exchange, asset = pair.split("-")

mat = load_matrix(DATAPATH / f"{pair}.csv.gz")

# Announcement / listing times
announcement_time = pd.to_datetime(listings_map[pair]["announced_at"], unit="ms")
listing_time      = pd.to_datetime(listings_map[pair]["trade_at"],     unit="ms")

# ---------------------------
# Presets
# ---------------------------
PRESETS = {
    "Announcement - 3h → Listing + 3h": (announcement_time - pd.Timedelta(hours=3),
                                         listing_time + pd.Timedelta(hours=3)),
    "Announcement → Listing + 3h":      (announcement_time,
                                         listing_time + pd.Timedelta(hours=3)),
    "Announcement - 3h → Listing":      (announcement_time - pd.Timedelta(hours=3),
                                         listing_time),
    "Listing - 3h → Listing + 3h":      (listing_time - pd.Timedelta(hours=3),
                                         listing_time + pd.Timedelta(hours=3)),
}
preset_label = st.selectbox("Preset", list(PRESETS.keys()))
win_from, win_to = PRESETS[preset_label]

# Clamp to matrix bounds (use overlap of index & columns as global bounds)
data_from = max(mat.index.min(), mat.columns.min())
data_to   = min(mat.index.max(), mat.columns.max())
win_from, win_to = clamp_window(win_from, win_to, data_from, data_to)
if win_from > win_to:
    st.warning("Selected preset has no overlap with data range.")
    st.stop()

# Filter to square window on both axes
mat_win = mat.loc[(mat.index >= win_from) & (mat.index <= win_to),
                  (mat.columns >= win_from) & (mat.columns <= win_to)]
if mat_win.empty:
    st.warning("No data in the selected window.")
    st.stop()

# ---------------------------
# Build figure
# ---------------------------
x = mat_win.columns.to_pydatetime().tolist()
y = mat_win.index.to_pydatetime().tolist()
z = mat_win.values

fig = go.Figure(go.Heatmap(
    z=z, x=x, y=y,
    colorscale="RdYlGn",
    zmid=0,
    colorbar=dict(title="Return (%)"),
    hovertemplate="Entry: %{y|%H:%M}<br>Exit: %{x|%H:%M}<br>Return: %{z:.2f}%<extra></extra>",
))

chart_height = 800
screen_height = streamlit_js_eval(js_expressions='screen.height', key='SCR_HEIGHT')
if screen_height:
    chart_height = screen_height * 0.6

chart_width = 1920
screen_width = streamlit_js_eval(js_expressions='screen.width', key='SCR_WIDTH')
if screen_width:
    chart_width = screen_width * 0.8

fig.update_layout(
    title=f"Profit/Return Heatmap (Entry vs Exit) — {exchange} {asset}<br><sup>{preset_label}</sup>",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    margin=dict(l=60, r=40, t=90, b=80),
    xaxis=dict(type="date", title="Exit time", tickformat="%H:%M", nticks=8),
    yaxis=dict(type="date", title="Entry time", tickformat="%H:%M", nticks=8, autorange="reversed"),
    height= chart_height,
)


# Reference lines (snap to filtered axes)
x_ann = nearest_index_val(mat_win.columns, announcement_time)
x_list = nearest_index_val(mat_win.columns, listing_time)
y_ann = nearest_index_val(mat_win.index,   announcement_time)
y_list = nearest_index_val(mat_win.index,   listing_time)

# Use legendgroup to dedupe legend (one entry per concept)
add_ref_line(fig, orientation="v", pos_dt=x_ann,  mat_win=mat_win,
             name="Announcement", color="deepskyblue", legendgroup="Announcement", showlegend=True)
add_ref_line(fig, orientation="h", pos_dt=y_ann,  mat_win=mat_win,
             name="Announcement", color="deepskyblue", legendgroup="Announcement", showlegend=False)
add_ref_line(fig, orientation="v", pos_dt=x_list, mat_win=mat_win,
             name="Listing",      color="limegreen",  legendgroup="Listing",      showlegend=True)
add_ref_line(fig, orientation="h", pos_dt=y_list, mat_win=mat_win,
             name="Listing",      color="limegreen",  legendgroup="Listing",      showlegend=False)

# Legend below-left
fig.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
)

# Watermark
wm_size = max(36, min(0.14 * chart_height, 120))

fig.add_annotation(
    x=0.5, y=0.5, xref="paper", yref="paper",
    text="DataMaxi+",
    showarrow=False,
    xanchor="center", yanchor="middle",
    font=dict(size=int(wm_size), color="white"),
    opacity=0.3  # subtle
)

png_bytes = fig.to_image(
    format="png",
    scale=2,
    )
st.download_button(
    "⬇️ Download chart (PNG)",
    data=png_bytes,
    file_name=f"{pair}.png",
    mime="image/png",
    use_container_width=True,
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True},)


st.markdown(
    'Data provided by <a href="https://datamaxiplus.com" target="_blank" style="color:#FFFFFF; text-decoration: none;">DataMaxi<span style="color:#F9D342;">+</span></a>',
    unsafe_allow_html=True,
)

