# ==============================================================
# F1 2024 | Driver Form & Team Strength vs Finishing Position (Per Race)
# ==============================================================

import fastf1
import pandas as pd
import numpy as np
import warnings, os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache(r"C:\\Users\\rohan\\Downloads\\Research STA199\\cache")

# === Team colors ==============================================
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

TEAM_CODE = {
    "Red Bull Racing": "RBR",
    "Ferrari": "FER",
    "Mercedes": "MER",
    "McLaren": "MCL",
    "Aston Martin": "AMR",
    "Alpine": "ALP",
    "RB": "RB",
    "Kick Sauber": "KSA",
    "Haas F1 Team": "HAA",
    "Williams": "WIL"
}

# === Name normalization =======================================
TEAM_RENAME = {
    "Alfa Romeo": "Kick Sauber",
    "Stake F1 Team": "Kick Sauber",
    "Sauber": "Kick Sauber",
    "AlphaTauri": "RB",
    "Scuderia AlphaTauri": "RB",
    "RB F1 Team": "RB"
}

# === Full GP list (2024 calendar order) =======================
full_2024_gps = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Japanese Grand Prix",
    "Chinese Grand Prix", "Miami Grand Prix", "Emilia Romagna Grand Prix", "Monaco Grand Prix",
    "Canadian Grand Prix", "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
    "Azerbaijan Grand Prix", "Singapore Grand Prix", "United States Grand Prix", "Mexican Grand Prix",
    "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
]

# ==============================================================
# Load race results
# ==============================================================
def load_race(year, gp):
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load(laps=False, telemetry=False, weather=False)

        df = session.results.copy()
        df = df[["Abbreviation", "TeamName", "GridPosition", "Position", "Points", "Status"]].rename(columns={
            "Abbreviation": "Driver",
            "TeamName": "Team",
            "GridPosition": "QualifyingPosition",
            "Position": "FinishingPosition"
        })
        df["Circuit"] = session.event["EventName"]
        df["Round"] = session.event["RoundNumber"]

        # Mark DNFs
        df["FinishingPosition"] = pd.to_numeric(df["FinishingPosition"], errors="coerce")
        df.loc[df["Status"].astype(str).str.contains("DNF|Ret|DSQ|Did not finish|Disqualified", case=False, na=False),
               "FinishingPosition"] = 21

        df["Team"] = df["Team"].replace(TEAM_RENAME)
        return df
    except Exception as e:
        print(f"⚠️ Failed to load {gp}: {e}")
        return pd.DataFrame()

# ==============================================================
# Load both seasons
# ==============================================================
df_2023 = pd.concat([load_race(2023, gp) for gp in full_2024_gps if not load_race(2023, gp).empty], ignore_index=True)
df_2024 = pd.concat([load_race(2024, gp) for gp in full_2024_gps if not load_race(2024, gp).empty], ignore_index=True)

print(f"✅ 2023 races: {df_2023['Circuit'].nunique()} | 2024 races: {df_2024['Circuit'].nunique()}")

# ==============================================================
# Prior-year reference (used for first race of 2024)
# ==============================================================
prev_driver_form = df_2023.groupby("Driver")["Points"].apply(lambda x: x.tail(3).mean()).to_dict()
prev_team_strength = df_2023.groupby("Team")["Points"].mean().to_dict()

# ==============================================================
# Combine seasons
# ==============================================================
df = pd.concat([df_2023.assign(Year=2023), df_2024.assign(Year=2024)], ignore_index=True)
df = df.sort_values(["Year", "Team", "Driver", "Round"]).reset_index(drop=True)
print(f"✅ Loaded {df['Circuit'].nunique()} circuits, {len(df)} entries total.")

# ==============================================================
# Compute Driver Form (avg points last 3 races)
# ==============================================================
def compute_driver_form(sub):
    driver = sub.name[1]
    sub = sub.sort_values(["Year", "Round"]).reset_index(drop=True)
    forms = []
    for i in range(len(sub)):
        if i == 0:
            forms.append(prev_driver_form.get(driver, np.nan))
        elif i == 1:
            vals = [sub.loc[i - 1, "Points"]]
            if driver in prev_driver_form: vals += [prev_driver_form[driver]] * 2
            forms.append(np.nanmean(vals))
        elif i == 2:
            vals = list(sub.loc[:i - 1, "Points"])
            if driver in prev_driver_form: vals += [prev_driver_form[driver]]
            forms.append(np.nanmean(vals[-3:]))
        else:
            forms.append(np.nanmean(sub.loc[i - 3:i - 1, "Points"]))
    sub["DriverForm"] = forms
    return sub

df = df.groupby(["Year", "Driver"], group_keys=False).apply(compute_driver_form)

# ==============================================================
# Compute Team Strength (rolling avg of last 3 team points)
# ==============================================================

team_strength = (
    df.groupby(["Team", "Round"], as_index=False)["Points"]
      .mean()
      .sort_values(["Team", "Round"])
)

# Use last 3 races rolling mean instead of expanding mean
team_strength["TeamStrength"] = (
    team_strength.groupby("Team")["Points"]
      .transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
)

# Merge back into main dataframe
df = df.merge(team_strength[["Team", "Round", "TeamStrength"]], on=["Team", "Round"], how="left")

# Fix Round 1 using previous year's averages
df.loc[df["Round"] == 1, "TeamStrength"] = df.loc[df["Round"] == 1, "Team"].map(prev_team_strength)

# ==============================================================
# Plotting helper (shared for both graphs)
# ==============================================================
def plot_per_race(df, x_col, x_label, title_prefix, save_path, hover_type="driver"):
    if "Year" in df.columns:
        df_2024 = df[df["Year"] == 2024]
    else:
        df_2024 = df  # already filtered or aggregated dataset

    ordered_circuits = (
        df_2024.drop_duplicates("Circuit")[["Circuit", "Round"]]
        .sort_values("Round")["Circuit"].tolist()
    )

    n_races = len(ordered_circuits)
    cols = 3
    rows = int(np.ceil(n_races / cols))

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=ordered_circuits)

    for i, circuit in enumerate(ordered_circuits):
        sub = df_2024[df_2024["Circuit"] == circuit]
        if sub.empty:
            continue
        row, col = (i // cols) + 1, (i % cols) + 1

        # === Scatter points ===
        for team in sub["Team"].dropna().unique():
            team_df = sub[sub["Team"] == team]
            if hover_type == "driver":
                hovertext = team_df.apply(
                    lambda r: f"{r.Driver} ({r.Team})<br>{x_label}: {r[x_col]:.2f}<br>Finish: {r.FinishingPosition}<br>Round: {r.Round}",
                    axis=1)
            else:
                hovertext = team_df.apply(
                    lambda r: f"{r.Team}<br>{x_label}: {r[x_col]:.2f}<br>Finish: {r.FinishingPosition:.1f}<br>Round: {r.Round}",
                    axis=1)
            x_jitter = team_df[x_col] + np.random.uniform(-0.12, 0.12, size=len(team_df))
            fig.add_trace(
                go.Scatter(
                    x=x_jitter,
                    y=team_df["FinishingPosition"],
                    mode="markers+text",
                    text=team_df["Driver"] if hover_type=="driver" else team_df["Team"].map(TEAM_CODE),
                    textposition="top center",
                    textfont=dict(size=9),

                    name=team,
                    hovertext=hovertext,
                    hoverinfo="text",

                    marker=dict(size=7, color=TEAM_COLORS.get(team, "gray")),
                    showlegend=False
                ),
                row=row, col=col
            )

        # === Expected descending trend line ===
        if sub[x_col].notna().sum() > 1:
            x_sorted = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
            y_trend = np.interp(x_sorted, [sub[x_col].min(), sub[x_col].max()], [20, 1])
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_trend,
                    mode="lines",
                    line=dict(color="black", width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=row, col=col
            )

        # DNF marker line
        fig.add_hline(y=21, line=dict(color="red", dash="dot"), row=row, col=col)

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text="Finishing Position", range=[0.5, 23], dtick=2)
    fig.update_layout(
        title_text=f"🏎️ F1 2024: {title_prefix} vs Finishing Position (Per Race, Ordered by Round)",
        template="plotly_white",
        height=4000, width=1800,
        font=dict(size=11),
        margin=dict(t=100, l=60, r=60, b=60)
    )
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"💾 Saved: {save_path}")

# ==============================================================
# 1️⃣ Driver Form Plot (per driver)
# ==============================================================
plot_per_race(
    df,
    x_col="DriverForm",
    x_label="Driver Form (Avg Points Last 3 Races)",
    title_prefix="Driver Form",
    save_path=r"C:\Users\rohan\Downloads\DriverForm_vs_Finish_PerRace.html",
    hover_type="driver"
)

# ==============================================================
# 2️⃣ Team Strength Plot (aggregated per team)
# ==============================================================
team_avg = (
    df[df["Year"] == 2024]
    .groupby(["Team", "Round", "Circuit"], as_index=False)
    .agg({
        "TeamStrength": "mean",
        "FinishingPosition": "mean"
    })
)
team_avg["FinishingPosition"] = team_avg["FinishingPosition"].round(1)
print(f"✅ Aggregated to {len(team_avg)} team-level records (~10 per race).")

plot_per_race(
    team_avg,
    x_col="TeamStrength",
    x_label="Team Strength (Rolling Avg Team Points)",
    title_prefix="Team Strength",
    save_path=r"C:\Users\rohan\Downloads\TeamStrength_vs_Finish_PerRace.html",
    hover_type="team"
)

print("✅ Generated both per-race grids successfully.")
