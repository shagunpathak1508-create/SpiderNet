"""
SpiderNet — Crime Intelligence Dashboard
Full Streamlit App: all notebook sections as interactive panels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🕷️ SpiderNet Crime Intelligence",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL DARK THEME STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #cdd9e5;
    font-family: 'Segoe UI', sans-serif;
  }
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  /* Cards / metric */
  [data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px; }
  [data-testid="stMetricValue"] { color: #cdd9e5 !important; font-size: 26px; font-weight: bold; }
  /* Section headers */
  h1 { color: #58a6ff !important; }
  h2 { color: #e6edf3 !important; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  h3 { color: #cdd9e5 !important; }
  /* Dividers */
  hr { border-color: #30363d; }
  /* Tabs */
  [data-testid="stTab"] { color: #8b949e; }
  button[data-baseweb="tab"] { color: #8b949e !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff; }
  /* Sidebar select */
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiselect label,
  [data-testid="stSidebar"] .stSlider label { color: #8b949e; }
  /* Alerts */
  .alert-red   { background:#2d1215; border-left:4px solid #ff6b6b; padding:12px 16px; border-radius:0 6px 6px 0; margin:6px 0; color:#ff6b6b; }
  .alert-blue  { background:#1a2433; border-left:4px solid #58a6ff; padding:12px 16px; border-radius:0 6px 6px 0; margin:6px 0; color:#58a6ff; }
  .alert-green { background:#12261e; border-left:4px solid #3fb950; padding:12px 16px; border-radius:0 6px 6px 0; margin:6px 0; color:#3fb950; }
  .badge-red   { background:#3d1e1e; color:#ff6b6b; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:bold; }
  .badge-blue  { background:#1e2e3d; color:#58a6ff; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:bold; }
  .badge-green { background:#1e3d2e; color:#3fb950; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB DARK STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "text.color": "#cdd9e5", "grid.color": "#21262d",
    "lines.color": "#58a6ff", "savefig.facecolor": "#0d1117",
})

# ─────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────
DATA_PATH = "data/train.csv"

@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()
    df = df.dropna(subset=["dates", "pddistrict"])
    df["category"]   = df["category"].str.strip().str.upper()
    df["pddistrict"] = df["pddistrict"].str.strip().str.upper()
    df["dayofweek"]  = df["dayofweek"].str.strip().str.title()
    df["resolution"] = df["resolution"].str.strip().str.upper()
    # GPS sanity
    df = df[(df["x"].between(-123.0, -122.3)) & (df["y"].between(37.6, 37.9))]
    # Dates
    df["dates"]      = pd.to_datetime(df["dates"])
    df["year"]       = df["dates"].dt.year
    df["month"]      = df["dates"].dt.month
    df["hour"]       = df["dates"].dt.hour
    df["week_start"] = df["dates"].dt.to_period("W").apply(lambda r: r.start_time)
    return df

@st.cache_data(show_spinner=False)
def compute_weekly(_df):
    weekly = (
        _df.groupby("week_start").size()
        .reset_index(name="crime_count")
        .sort_values("week_start").reset_index(drop=True)
    )
    weekly["prev_week"]  = weekly["crime_count"].shift(1)
    weekly["delta"]      = weekly["crime_count"] - weekly["prev_week"]
    weekly["pct_change"] = (weekly["delta"] / weekly["prev_week"] * 100).round(2)
    weekly["z_score"]    = stats.zscore(weekly["crime_count"].fillna(0))
    weekly["is_spike"]   = weekly["z_score"].abs() > 2
    weekly["spike_type"] = np.where(
        weekly["z_score"] > 2, "HIGH",
        np.where(weekly["z_score"] < -2, "LOW", "NORMAL")
    )
    return weekly

@st.cache_data(show_spinner=False)
def compute_districts(_df):
    d = (
        _df.groupby("pddistrict")
        .agg(total=("category","count"), types=("category","nunique"),
             lat=("y","mean"), lon=("x","mean"))
        .reset_index().sort_values("total", ascending=False).reset_index(drop=True)
    )
    d["rank"]    = range(1, len(d)+1)
    d["pct"]     = (d["total"] / d["total"].sum() * 100).round(2)
    return d

@st.cache_data(show_spinner=False)
def run_kmeans(_df, k=8, n=40_000):
    sample = _df[["x","y"]].dropna().sample(min(n, len(_df)), random_state=42)
    sc   = StandardScaler()
    X    = sc.fit_transform(sample.values)
    km   = KMeans(n_clusters=k, random_state=42, n_init=10)
    sample = sample.copy()
    sample["cluster"] = km.fit_predict(X)
    centroids = sc.inverse_transform(km.cluster_centers_)
    return sample, centroids

# ─────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕷️ SpiderNet")
    st.markdown("*Crime Intelligence Platform*")
    st.markdown("---")

    with st.spinner("Loading data…"):
        df_raw = load_and_clean()

    years = sorted(df_raw["year"].unique())
    sel_years = st.multiselect("📅 Year filter", years, default=years)

    all_cats = sorted(df_raw["category"].unique())
    sel_cats = st.multiselect("🏷️ Crime category", all_cats,
                              placeholder="All categories (default)")

    all_dists = sorted(df_raw["pddistrict"].unique())
    sel_dists = st.multiselect("📍 District", all_dists,
                               placeholder="All districts (default)")

    spike_thresh = st.slider("Spike Z-score threshold", 1.0, 4.0, 2.0, 0.1)

    st.markdown("---")
    st.caption(f"Dataset: **{len(df_raw):,}** records")
    st.caption(f"{df_raw['dates'].min().date()} → {df_raw['dates'].max().date()}")

# Apply filters
df = df_raw.copy()
if sel_years:
    df = df[df["year"].isin(sel_years)]
if sel_cats:
    df = df[df["category"].isin(sel_cats)]
if sel_dists:
    df = df[df["pddistrict"].isin(sel_dists)]

weekly     = compute_weekly(df)
weekly["is_spike"] = weekly["z_score"].abs() > spike_thresh
weekly["spike_type"] = np.where(
    weekly["z_score"] > spike_thresh, "HIGH",
    np.where(weekly["z_score"] < -spike_thresh, "LOW", "NORMAL")
)
districts  = compute_districts(df)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🕷️ SpiderNet — Crime Intelligence Dashboard")
st.markdown("*San Francisco Police Department Crime Data — Powered by SpiderNet*")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("📊 Total Crimes",         f"{len(df):,}")
k2.metric("📅 Weeks Tracked",         f"{len(weekly):,}")
k3.metric("🔴 Anomalous Weeks",       f"{weekly['is_spike'].sum()}")
k4.metric("📍 Top District",          districts.iloc[0]["pddistrict"])
k5.metric("🏷️ Top Category",         df["category"].value_counts().index[0].split("/")[0].strip())
k6.metric("⏰ Peak Hour",             f"{df['hour'].value_counts().index[0]:02d}:00")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Weekly Spikes",
    "🗺️ District Map",
    "📊 Breakdown",
    "🤖 Clusters",
    "🔍 Explainer",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — WEEKLY SPIKE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("Weekly Crime Volume — Spike Detection")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Weekly Crimes",  f"{weekly['crime_count'].mean():,.0f}")
    c2.metric("Max Single Week",    f"{weekly['crime_count'].max():,}")
    c3.metric("HIGH Spikes",        f"{(weekly['z_score'] > spike_thresh).sum()}",
              delta=f"z > {spike_thresh}")

    # Main chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios":[2,1]})

    # Weekly line
    ax1.plot(weekly["week_start"], weekly["crime_count"],
             color="#58a6ff", linewidth=1.4, label="Weekly crimes")
    ax1.fill_between(weekly["week_start"], weekly["crime_count"],
                     alpha=0.12, color="#58a6ff")

    rolling = weekly["crime_count"].rolling(4, center=True).mean()
    ax1.plot(weekly["week_start"], rolling, "--", color="#f0db4f",
             linewidth=1.1, alpha=0.85, label="4-wk rolling avg")

    # Spike markers
    hi = weekly[weekly["spike_type"] == "HIGH"]
    lo = weekly[weekly["spike_type"] == "LOW"]
    ax1.scatter(hi["week_start"], hi["crime_count"], color="#ff6b6b",
                s=70, zorder=5, label="HIGH spike", marker="^")
    ax1.scatter(lo["week_start"], lo["crime_count"], color="#4ecdc4",
                s=70, zorder=5, label="LOW spike", marker="v")

    ax1.set_title("Weekly Crime Counts with Anomaly Detection", fontsize=13, color="#cdd9e5")
    ax1.set_ylabel("Crime Count")
    ax1.legend(facecolor="#1c2128", labelcolor="#cdd9e5", framealpha=0.9)
    ax1.grid(True, alpha=0.15)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25)

    # % change bars
    colors = ["#ff6b6b" if v > 0 else "#4ecdc4"
              for v in weekly["pct_change"].fillna(0)]
    ax2.bar(weekly["week_start"], weekly["pct_change"].fillna(0),
            color=colors, width=5, alpha=0.8)
    ax2.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax2.set_title("Week-over-Week % Change", fontsize=11, color="#cdd9e5")
    ax2.set_ylabel("% Change")
    ax2.grid(True, alpha=0.1)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Spike table
    st.subheader("🔴 Detected Spike Weeks")
    spikes_df = weekly[weekly["is_spike"]].sort_values("z_score", key=abs, ascending=False)
    if not spikes_df.empty:
        display_cols = ["week_start","crime_count","prev_week","pct_change","z_score","spike_type"]
        styled = spikes_df[display_cols].rename(columns={
            "week_start":"Week", "crime_count":"Crimes",
            "prev_week":"Prev Week", "pct_change":"% Change",
            "z_score":"Z-Score", "spike_type":"Type"
        })
        def color_type(val):
            if val == "HIGH":   return "background-color:#2d1215; color:#ff6b6b"
            if val == "LOW":    return "background-color:#1a2433; color:#4ecdc4"
            return ""
        st.dataframe(
            styled.style.applymap(color_type, subset=["Type"]),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No anomalous weeks found with current filter/threshold.")

    # Download
    st.download_button(
        "⬇️ Download Weekly Summary CSV",
        weekly.to_csv(index=False).encode(),
        "weekly_crime_summary.csv",
        "text/csv"
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — DISTRICT MAP  (pulse + web)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("District Crime Map — Pulse Spikes · Crime Web")

    map_col, rank_col = st.columns([3, 1])

    with rank_col:
        st.markdown("**🏆 District Ranking**")
        for _, row in districts.iterrows():
            badge = "badge-red" if row["rank"] <= 3 else ("badge-blue" if row["rank"] <= 7 else "badge-green")
            emoji = "🔴" if row["rank"] <= 3 else ("🟡" if row["rank"] <= 6 else "🟢")
            st.markdown(
                f"{emoji} <span class='{badge}'>#{row['rank']}</span> &nbsp;"
                f"**{row['pddistrict']}**<br>"
                f"<small style='color:#8b949e'>{row['total']:,} crimes &nbsp;({row['pct']}%)</small>",
                unsafe_allow_html=True
            )
            st.markdown("")
        st.markdown("---")
        st.markdown("<small style='color:#8b949e'>🔴 Pulse ring = HIGH crime<br>🕸️ Lines = shared crime type</small>",
                    unsafe_allow_html=True)

    with map_col:
        MAX_R, MIN_R = 40, 8
        max_t = districts["total"].max()
        min_t = districts["total"].min()
        # z-score across districts to flag spikes
        dist_z = stats.zscore(districts["total"])
        spike_districts = set(districts.loc[dist_z > 0.5, "pddistrict"].tolist())

        def get_color(rank, n=len(districts)):
            ratio = 1 - (rank - 1) / max(n - 1, 1)
            r = int(255 * ratio); g = int(255 * (1 - ratio))
            return f"#{r:02x}{g:02x}40"

        # ── build map ──────────────────────────────────────────
        m = folium.Map(location=[37.77, -122.42], zoom_start=12,
                       tiles="CartoDB dark_matter")

        # ── inject global CSS for pulse keyframes ───────────────
        pulse_css = """
        <style>
        @keyframes spiderpulse {
            0%   { transform: scale(1);   opacity: 0.9; }
            50%  { transform: scale(1.55); opacity: 0.4; }
            100% { transform: scale(1);   opacity: 0.9; }
        }
        @keyframes spiderring {
            0%   { transform: scale(1);   opacity: 0.7; box-shadow: 0 0 0px 0px #ff6b6b; }
            50%  { transform: scale(1.7); opacity: 0.0; box-shadow: 0 0 12px 8px #ff6b6b; }
            100% { transform: scale(1);   opacity: 0.7; box-shadow: 0 0 0px 0px #ff6b6b; }
        }
        .pulse-ring {
            border-radius: 50%;
            background: rgba(255,107,107,0.25);
            border: 2px solid #ff6b6b;
            animation: spiderring 1.8s ease-in-out infinite;
            pointer-events: none;
        }
        .pulse-ring-mid {
            border-radius: 50%;
            background: rgba(255,107,107,0.12);
            border: 1.5px dashed #ff6b6b;
            animation: spiderring 1.8s ease-in-out infinite 0.5s;
            pointer-events: none;
        }
        .pulse-dot {
            border-radius: 50%;
            background: #ff6b6b;
            animation: spiderpulse 1.8s ease-in-out infinite;
            pointer-events: none;
        }
        </style>
        """
        m.get_root().html.add_child(folium.Element(pulse_css))

        # ── per-district circles + pulse rings ──────────────────
        for _, row in districts.iterrows():
            radius  = MIN_R + (row["total"] - min_t) / max(max_t - min_t, 1) * (MAX_R - MIN_R)
            col     = get_color(row["rank"])
            is_spike = row["pddistrict"] in spike_districts
            pop = (
                f"<div style='font-family:monospace;background:#0d1117;color:#cdd9e5;"
                f"padding:10px;border-radius:8px;min-width:180px'>"
                f"<b style='color:#58a6ff'>#{row['rank']} {row['pddistrict']}</b><br>"
                f"Crimes: <b style='color:#ff6b6b'>{row['total']:,}</b><br>"
                f"Share: {row['pct']}%<br>Crime types: {row['types']}"
                + ("<br><b style='color:#ff6b6b'>⚠ HIGH CRIME ZONE</b>" if is_spike else "") +
                "</div>"
            )

            # Base filled circle
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius, color=col,
                fill=True, fill_color=col, fill_opacity=0.75,
                popup=folium.Popup(pop, max_width=240),
                tooltip=f"{row['pddistrict']}: {row['total']:,} (#{row['rank']})"
            ).add_to(m)

            # Animated pulse ring only on spike districts
            if is_spike:
                px = int(radius * 3.5)   # pixel size for the div
                html_div = f"""
                <div style="position:relative;width:{px}px;height:{px}px;">
                  <div class="pulse-ring-mid" style="position:absolute;
                       width:{px}px;height:{px}px;top:0;left:0;"></div>
                  <div class="pulse-ring" style="position:absolute;
                       width:{int(px*0.72)}px;height:{int(px*0.72)}px;
                       top:{int(px*0.14)}px;left:{int(px*0.14)}px;"></div>
                  <div class="pulse-dot" style="position:absolute;
                       width:{int(px*0.28)}px;height:{int(px*0.28)}px;
                       top:{int(px*0.36)}px;left:{int(px*0.36)}px;"></div>
                </div>"""
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    icon=folium.DivIcon(html=html_div,
                                        icon_size=(px, px),
                                        icon_anchor=(px//2, px//2))
                ).add_to(m)

        # ── CRIME WEB: PolyLines between districts per top category ──
        # For each top-5 category, find 2+ districts where it ranks #1 or #2
        # and draw connecting lines (the "spider web")
        WEB_COLORS = {
            0: "#58a6ff",  # blue
            1: "#ff6b6b",  # red
            2: "#3fb950",  # green
            3: "#f0db4f",  # yellow
            4: "#d2a8ff",  # purple
        }
        # top crime category per district
        dist_top_cat = (
            df.groupby(["pddistrict","category"])
            .size().reset_index(name="n")
        )
        dist_top_cat = dist_top_cat.sort_values("n", ascending=False)
        dist_top1 = dist_top_cat.groupby("pddistrict").first().reset_index()

        top5_web_cats = dist_top1["category"].value_counts().head(5).index.tolist()
        dist_coords   = districts.set_index("pddistrict")[["lat","lon"]].to_dict("index")

        for ci, cat in enumerate(top5_web_cats):
            members = dist_top1[dist_top1["category"] == cat]["pddistrict"].tolist()
            members = [d for d in members if d in dist_coords]
            if len(members) < 2:
                continue
            web_col = WEB_COLORS.get(ci, "#ffffff")
            # connect every pair  (spider-web style)
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    a = dist_coords[members[i]]
                    b = dist_coords[members[j]]
                    folium.PolyLine(
                        locations=[[a["lat"], a["lon"]], [b["lat"], b["lon"]]],
                        color=web_col,
                        weight=1.6,
                        opacity=0.55,
                        dash_array="6 4",
                        tooltip=f"🕸️ Shared: {cat}"
                    ).add_to(m)

        # Web legend
        legend_html = (
            "<div style='position:fixed;bottom:30px;left:60px;z-index:9999;"
            "background:rgba(13,17,23,0.92);border:1px solid #30363d;"
            "border-radius:8px;padding:10px 14px;font-family:monospace;"
            "font-size:12px;color:#cdd9e5;min-width:200px;'>"
            "<b style='color:#58a6ff'>🕸️ Crime Web Legend</b><br>"
        )
        for ci, cat in enumerate(top5_web_cats):
            wc = WEB_COLORS.get(ci, "#fff")
            legend_html += (
                f"<span style='color:{wc}'>━━</span> "
                f"{cat[:22]}<br>"
            )
        legend_html += (
            "<hr style='border-color:#30363d;margin:6px 0'>"
            "<span style='color:#ff6b6b'>● pulse</span> = HIGH crime district"
            "</div>"
        )
        m.get_root().html.add_child(folium.Element(legend_html))

        # Heatmap overlay toggle
        show_heat = st.checkbox("Show heatmap overlay", value=False)
        if show_heat:
            heat_sample = df[["y","x"]].dropna().sample(min(20_000, len(df)), random_state=1)
            HeatMap(heat_sample.values.tolist(), radius=12, blur=16).add_to(m)

        st_folium(m, width=None, height=560, returned_objects=[])

    st.download_button(
        "⬇️ Download District Ranking CSV",
        districts.to_csv(index=False).encode(),
        "district_ranking.csv", "text/csv"
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — BREAKDOWN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("Crime Breakdown Analysis")

    row1a, row1b = st.columns(2)

    # Category chart
    with row1a:
        top_n = st.slider("Top N categories", 5, 20, 10)
        top_cats = df["category"].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(7, 4))
        pal = plt.cm.plasma(np.linspace(0.9, 0.3, len(top_cats)))
        bars = ax.barh(top_cats.index[::-1], top_cats.values[::-1], color=pal[::-1])
        ax.set_title(f"Top {top_n} Crime Categories", color="#cdd9e5", fontsize=12)
        ax.set_xlabel("Count")
        ax.grid(True, axis="x", alpha=0.15)
        for b, v in zip(bars, top_cats.values[::-1]):
            ax.text(b.get_width() + 100, b.get_y() + b.get_height()/2,
                    f"{v:,}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Day-of-week distribution
    with row1b:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_counts = df["dayofweek"].value_counts().reindex(day_order).fillna(0)
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_cols = plt.cm.cool(np.linspace(0.3, 0.9, 7))
        ax.bar(day_counts.index, day_counts.values, color=bar_cols)
        ax.set_title("Crime Count by Day of Week", color="#cdd9e5", fontsize=12)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.15)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Day × Hour heatmap
    st.subheader("🕐 Day × Hour Crime Heatmap")
    pivot = df.pivot_table(index="dayofweek", columns="hour",
                            values="category", aggfunc="count", observed=True)
    pivot = pivot.reindex(day_order)
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
                linecolor="#0d1117", cbar_kws={"label": "Crime Count"})
    ax.set_title("Crime Frequency — Day of Week × Hour of Day",
                 color="#cdd9e5", fontsize=12)
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Yearly trend
    st.subheader("📆 Yearly Crime Trend")
    yearly = df.groupby(["year","category"]).size().reset_index(name="count")
    top5_cats = df["category"].value_counts().head(5).index.tolist()
    yearly_top = yearly[yearly["category"].isin(top5_cats)]
    fig, ax = plt.subplots(figsize=(12, 4))
    pal5 = ["#58a6ff","#ff6b6b","#3fb950","#f0db4f","#d2a8ff"]
    for i, cat in enumerate(top5_cats):
        sub = yearly_top[yearly_top["category"] == cat]
        ax.plot(sub["year"], sub["count"], "o-", color=pal5[i], label=cat[:20], linewidth=1.8)
    ax.set_title("Top 5 Crime Categories — Year over Year", color="#cdd9e5", fontsize=12)
    ax.set_xlabel("Year"); ax.set_ylabel("Count")
    ax.legend(facecolor="#1c2128", labelcolor="#cdd9e5", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — CLUSTER ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.subheader("🤖 Geographical Crime Hotspot Clusters (KMeans)")

    k_val = st.slider("Number of clusters (K)", 3, 15, 8)

    with st.spinner(f"Running KMeans K={k_val}…"):
        sample_km, centroids = run_kmeans(df, k=k_val)

    cl1, cl2 = st.columns(2)

    with cl1:
        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.cm.get_cmap("tab10" if k_val <= 10 else "hsv", k_val)
        for c in range(k_val):
            mask = sample_km["cluster"] == c
            ax.scatter(sample_km.loc[mask, "x"], sample_km.loc[mask, "y"],
                       s=1.5, alpha=0.35, color=cmap(c))
        ax.scatter(centroids[:,0], centroids[:,1], s=180, marker="*",
                   c="white", edgecolors="#0d1117", zorder=10, linewidths=0.5)
        ax.set_title(f"KMeans Crime Hotspots (K={k_val})",
                     color="#cdd9e5", fontsize=12)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.1)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with cl2:
        st.markdown("**Cluster Sizes**")
        cluster_counts = sample_km["cluster"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors2 = [cmap(i) for i in range(k_val)]
        ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors2)
        ax2.set_title("Incidents per Cluster", color="#cdd9e5", fontsize=11)
        ax2.set_xlabel("Cluster ID"); ax2.set_ylabel("Count (sample)")
        ax2.grid(True, axis="y", alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown("**Centroid Coordinates**")
        centroid_df = pd.DataFrame(centroids, columns=["Longitude","Latitude"])
        centroid_df.index.name = "Cluster"
        st.dataframe(centroid_df.round(5), use_container_width=True)

    # Folium cluster map
    st.subheader("🗺️ Interactive Cluster Map")
    cm_map = folium.Map(location=[37.77, -122.42], zoom_start=12,
                        tiles="CartoDB dark_matter")
    cmap_hex = ["#58a6ff","#ff6b6b","#3fb950","#f0db4f","#d2a8ff",
                "#ffa657","#ff7b72","#a5d6ff","#7ee787","#e3b341",
                "#f0883e","#56d364","#bc8cff","#79c0ff","#ff8c94"]
    plot_sample = sample_km.sample(min(8_000, len(sample_km)), random_state=7)
    for _, row in plot_sample.iterrows():
        col_hex = cmap_hex[int(row["cluster"]) % len(cmap_hex)]
        folium.CircleMarker(
            location=[row["y"], row["x"]],
            radius=2, color=col_hex, fill=True, fill_opacity=0.5,
            weight=0
        ).add_to(cm_map)
    for i, (cx, cy) in enumerate(centroids):
        col_hex = cmap_hex[i % len(cmap_hex)]
        folium.Marker(
            location=[cy, cx],
            popup=f"Cluster {i} centroid",
            icon=folium.Icon(color="white", icon="star", prefix="fa")
        ).add_to(cm_map)
    st_folium(cm_map, width=None, height=480, returned_objects=[])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — EXPLAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.subheader("🔍 Explainable Intelligence Panel")

    # --- Key facts
    top_dist       = districts.iloc[0]
    top_cat        = df["category"].value_counts()
    peak_hr        = df["hour"].value_counts().index[0]
    spike_hi       = weekly[weekly["spike_type"] == "HIGH"]
    worst_spike    = spike_hi.loc[spike_hi["pct_change"].idxmax()] if not spike_hi.empty else None
    spike_info     = "—"
    if worst_spike is not None:
        wkstr = pd.Timestamp(worst_spike["week_start"]).strftime("%d %b %Y")
        spike_info = f"Week of **{wkstr}** — +{worst_spike['pct_change']:.1f}%  (z = {worst_spike['z_score']:.2f})"

    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown("#### 📍 District Intelligence")
        for _, row in districts.head(5).iterrows():
            badge = "🔴" if row["rank"] == 1 else ("🟠" if row["rank"] <= 3 else "🟡")
            st.markdown(
                f"{badge} **{row['pddistrict']}** — {row['total']:,} crimes "
                f"({row['pct']}% of total) — {row['types']} unique crime types"
            )

        st.markdown("#### 🏷️ Top Crime Categories")
        for cat, cnt in top_cat.head(5).items():
            pct = cnt / len(df) * 100
            st.progress(int(pct), text=f"{cat}: {cnt:,} ({pct:.1f}%)")

    with ex2:
        st.markdown("#### 📈 Spike Summary")
        st.markdown(f"- Total anomalous weeks: **{weekly['is_spike'].sum()}**")
        st.markdown(f"- HIGH spikes: **{(weekly['z_score'] > spike_thresh).sum()}**")
        st.markdown(f"- LOW spikes: **{(weekly['z_score'] < -spike_thresh).sum()}**")
        st.markdown(f"- Worst spike: {spike_info}")

        st.markdown("#### ⏰ Temporal Patterns")
        st.markdown(f"- Peak crime hour: **{peak_hr:02d}:00**")
        st.markdown(f"- Peak crime day: **{df['dayofweek'].value_counts().index[0]}**")
        st.markdown(f"- Average weekly crimes: **{weekly['crime_count'].mean():,.0f}**")
        st.markdown(f"- Max weekly crimes: **{weekly['crime_count'].max():,}**")

    st.markdown("---")
    st.markdown("#### 💡 Recommended Actions")
    st.markdown(
        f"""
<div class='alert-red'>🚨 <b>Highest Risk District:</b> {top_dist['pddistrict']} — deploy additional patrol resources, 
especially during the <b>{peak_hr:02d}:00–{(peak_hr+2)%24:02d}:00</b> peak window.</div>

<div class='alert-blue'>📊 <b>Monitor:</b> Trigger alert when weekly crime z-score exceeds {spike_thresh:.1f}. 
Current detection found <b>{weekly['is_spike'].sum()} anomalous weeks</b> in the dataset.</div>

<div class='alert-green'>🔄 <b>Strategic:</b> Re-evaluate cluster-based patrol zones quarterly. 
Focus resources on the <b>{top_cat.index[0]}</b> crime type (most frequent).</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### 📥 Raw Data Explorer")
    with st.expander("Browse filtered dataset"):
        page_size = 500
        st.dataframe(df.head(page_size)[[
            "dates","category","pddistrict","dayofweek","hour","resolution","address"
        ]], use_container_width=True)
        st.caption(f"Showing first {page_size:,} of {len(df):,} rows")

    st.download_button(
        "⬇️ Download Filtered Data CSV",
        df.to_csv(index=False).encode(),
        "spidernet_filtered.csv", "text/csv"
    )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6e7681;font-size:12px'>"
    "🕷️ SpiderNet Crime Intelligence Platform &nbsp;|&nbsp; "
    "SF Crime Dataset &nbsp;|&nbsp; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)