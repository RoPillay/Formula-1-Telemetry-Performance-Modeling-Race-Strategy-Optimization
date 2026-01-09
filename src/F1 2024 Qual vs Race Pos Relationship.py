# ==============================================================
# F1 2024 | Qualifying vs Finishing Position (All Races, Ordered & Expanded Axes)
# ==============================================================

import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache(r"C:\\Users\\rohan\\Downloads\\Research STA199\\cache")

TEAM_COLORS = {
    "Red Bull Racing": "#1E5BC6",
    "Ferrari": "#ED1C24",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "RB": "#2B4562",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Williams": "#005AFF"
}

# === Official 2024 GP list in chronological order ==============
full_2024_gps = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Japanese Grand Prix",
    "Chinese Grand Prix", "Miami Grand Prix", "Emilia Romagna Grand Prix", "Monaco Grand Prix",
    "Canadian Grand Prix", "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
    "Azerbaijan Grand Prix", "Singapore Grand Prix", "United States Grand Prix", "Mexican Grand Prix",
    "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
]

# === Load single race ==========================================
def load_race(year, gp):
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load(laps=False, telemetry=False, weather=False)

        res = session.results.copy()
        df = res[["Abbreviation", "TeamName", "GridPosition", "Position", "Status"]].rename(columns={
            "Abbreviation": "Driver",
            "TeamName": "Team",
            "GridPosition": "QualifyingPosition",
            "Position": "FinishingPosition"
        })

        df["Circuit"] = session.event["EventName"]
        df["Round"] = session.event["RoundNumber"]

        # Mark DNFs
        df["FinishingPosition"] = pd.to_numeric(df["FinishingPosition"], errors="coerce")
        df.loc[
            df["Status"].astype(str).str.contains("DNF|Ret|DSQ|Did not finish|Disqualified", case=False, na=False),
            "FinishingPosition"
        ] = 21

        # Skip incomplete
        if df.shape[0] < 10:
            print(f"⚠️ Skipping incomplete data for {gp}")
            return pd.DataFrame()

        print(f"✅ Loaded {gp} ({len(df)} entries)")
        return df
    except Exception as e:
        print(f"⚠️ Failed to load {gp}: {e}")
        return pd.DataFrame()

# === Load all 2024 races ======================================
year = 2024
all_races = [load_race(year, gp) for gp in full_2024_gps]
df = pd.concat([r for r in all_races if not r.empty], ignore_index=True)

print(f"\n✅ Loaded {df['Circuit'].nunique()} circuits, {len(df)} entries total.")

# === Clean numeric + delta ====================================
df = df.dropna(subset=["QualifyingPosition", "FinishingPosition"])
df["QualifyingPosition"] = df["QualifyingPosition"].astype(int)
df["FinishingPosition"] = df["FinishingPosition"].astype(int)
df["DeltaPos"] = df["FinishingPosition"] - df["QualifyingPosition"]

# === Correlation summary ======================================
slope, intercept, r_value, p_value, _ = stats.linregress(df["QualifyingPosition"], df["FinishingPosition"])
print(f"\n📊 Overall correlation: R² = {r_value**2:.3f}, p-value = {p_value:.5f}")

# === Ordered by Round =========================================
df = df.sort_values("Round").reset_index(drop=True)
ordered_circuits = df.drop_duplicates("Circuit")[["Circuit", "Round"]].sort_values("Round")["Circuit"].tolist()

# === Plot setup ===============================================
n_races = len(ordered_circuits)
cols = 3
rows = int(np.ceil(n_races / cols))

fig = make_subplots(
    rows=rows, cols=cols,
    subplot_titles=ordered_circuits,
    shared_xaxes=False, shared_yaxes=False
)

# === Add each subplot =========================================
for i, circuit in enumerate(ordered_circuits):
    sub = df[df["Circuit"] == circuit]
    if sub.empty:
        continue

    row = (i // cols) + 1
    col = (i % cols) + 1

    # Perfect finish line
    line_x = np.arange(0.5, 21.5)
    line_y = line_x

    # Scatter
    for idx, rowdata in sub.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[rowdata["QualifyingPosition"]],
                y=[rowdata["FinishingPosition"]],
                mode="markers+text",
                text=[rowdata["Driver"]],          # Driver abbreviation on the point
                textposition="top center",
                textfont=dict(size=9),

                hovertext=f"{rowdata.Driver} ({rowdata.Team})<br>"
                        f"Q: {rowdata.QualifyingPosition}, F: {rowdata.FinishingPosition}<br>"
                        f"Δ={rowdata.DeltaPos:+}",
                hoverinfo="text",

                marker=dict(size=8, 
                            color=TEAM_COLORS.get(rowdata["Team"], "gray"))
            ),
            row=row, col=col
        )

    # Add perfect line
    fig.add_trace(
        go.Scatter(x=line_x, y=line_y, mode="lines",
                   line=dict(color="black", dash="dash"),
                   name="Perfect y=x", showlegend=False),
        row=row, col=col
    )

    # Add DNF reference line
    fig.add_hline(y=21, line=dict(color="red", dash="dot"), row=row, col=col)

# === Layout + axis cleanup ====================================
fig.update_xaxes(
    title="Qualifying Position",
    range=[0.5, 21.5], dtick=2,
    tickfont=dict(size=9), title_font=dict(size=11)
)
fig.update_yaxes(
    title="Finishing Position",
    range=[0.5, 22.5], dtick=2,
    tickfont=dict(size=9), title_font=dict(size=11)
)
fig.update_layout(
    title_text="🏎️ F1 2024: Qualifying vs Finishing Position (Ordered by Round)",
    template="plotly_white",
    height=4000, width=1800,
    font=dict(size=11),
    showlegend=False,
    margin=dict(t=100, l=60, r=60, b=60)
)

# === Export ===================================================
output_path = r"C:\Users\rohan\Downloads\F1_2024_Qual_vs_Finish_Ordered.html"
fig.write_html(output_path, include_plotlyjs="cdn")

print(f"\n💾 Saved interactive file to:\n{output_path}")
print("Open directly in your browser — now ordered and unclipped.")
