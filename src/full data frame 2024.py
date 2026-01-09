# ==============================================================
#  F1 2024 | Full Dataset Builder (Dashboard-Compatible + Forced 24 Races)
# ==============================================================

import fastf1
import pandas as pd
import numpy as np
import os, warnings

warnings.filterwarnings("ignore")
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache(r"C:\\Users\\rohan\\Downloads\\Research STA199\\cache")

# === Force full 2024 schedule (24 rounds) =====================
full_2024_gps = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Japanese Grand Prix",
    "Chinese Grand Prix", "Miami Grand Prix", "Emilia Romagna Grand Prix", "Monaco Grand Prix",
    "Canadian Grand Prix", "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
    "Azerbaijan Grand Prix", "Singapore Grand Prix", "United States Grand Prix", "Mexican Grand Prix",
    "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
]
schedule = pd.DataFrame({
    "EventName": full_2024_gps,
    "RoundNumber": range(1, len(full_2024_gps) + 1),
    "EventFormat": ["conventional"] * len(full_2024_gps)
})

# === Load single race (dashboard-style) =======================
def load_race_data(year, gp):
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load()  # same lazy-load as dashboard

        res = getattr(session, "results", pd.DataFrame())
        if not res.empty:
            results = res[["Abbreviation", "TeamName", "GridPosition", "Position", "Points", "Status"]].copy()
        else:
            results = pd.DataFrame(columns=["Abbreviation","TeamName","GridPosition","Position","Points","Status"])

        results.rename(columns={
            "Abbreviation": "Driver",
            "TeamName": "Team",
            "GridPosition": "QualifyingPosition",
            "Position": "RacePosition"
        }, inplace=True)
        results["DNF"] = results["Status"].apply(lambda x: 1 if any(t in str(x) for t in ["Ret","DNF","DSQ"]) else 0)

        # Lap summary if exists
        laps = getattr(session, "laps", pd.DataFrame())
        if not laps.empty:
            laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
            lap_summary = laps.groupby("Driver")["LapTime_s"].agg(
                AvgLapTime="mean", FastestLapTime="min"
            ).reset_index()
            results = results.merge(lap_summary, on="Driver", how="left")
        else:
            results["AvgLapTime"] = np.nan
            results["FastestLapTime"] = np.nan

        # Weather if exists
        w = getattr(session, "weather_data", pd.DataFrame())
        if not w.empty:
            results["WeatherTemp"] = w["AirTemp"].mean()
            results["WeatherHumidity"] = w["Humidity"].mean()
            results["WeatherTempVar"] = w["AirTemp"].var()
            results["WeatherHumidityVar"] = w["Humidity"].var()
        else:
            for c in ["WeatherTemp","WeatherHumidity","WeatherTempVar","WeatherHumidityVar"]:
                results[c] = np.nan

        results["Circuit"] = session.event["EventName"]
        results["Year"] = year
        results["Round"] = session.event["RoundNumber"]
        results["NumLaps"] = getattr(session, "total_laps", np.nan)
        results["SprintWeekend"] = int(session.event["EventFormat"] == "sprint")

        return results

    except Exception as e:
        print(f"⚠️ Fallback placeholder for {gp} {year}: {e}")
        return pd.DataFrame()

# === Load full season (force all 24 races) =====================
def load_full_season(year):
    all_data, failed = [], []

    default_drivers = [
        "VER","PER","LEC","SAI","NOR","PIA","HAM","RUS","ALO","STR",
        "RIC","TSU","ALB","SAR","HUL","MAG","OCO","GAS","BOT","ZHO"
    ]

    for _, ev in schedule.iterrows():
        gp = ev["EventName"]
        df_gp = load_race_data(year, gp)

        if df_gp.empty or len(df_gp) < 20:
            failed.append(gp)
            df_gp = pd.DataFrame({
                "Driver": default_drivers,
                "Team": np.nan,
                "QualifyingPosition": np.nan,
                "RacePosition": np.arange(1, 21),
                "Points": np.nan, "Status": np.nan, "DNF": np.nan,
                "AvgLapTime": np.nan, "FastestLapTime": np.nan,
                "WeatherTemp": np.nan, "WeatherHumidity": np.nan,
                "WeatherTempVar": np.nan, "WeatherHumidityVar": np.nan,
                "Circuit": gp, "Year": year, "Round": ev["RoundNumber"],
                "NumLaps": np.nan, "SprintWeekend": int(ev["EventFormat"] == "sprint")
            })
        all_data.append(df_gp)

    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Loaded {len(df['Circuit'].unique())} circuits, {len(df)} driver entries total.")
    if failed:
        print(f"⚠️ Used placeholders for: {', '.join(failed)}")
    return df

# === Load 2023 for seeding ====================================
df_2023 = load_full_season(2023)
prev_driver_form = df_2023.groupby("Driver")["RacePosition"].apply(lambda x: x.tail(3).mean()).to_dict()
prev_team_strength = df_2023.groupby("Team")["Points"].mean().to_dict()
print("\n✅ Loaded 2023 season context for seeding.")

# === Load 2024 ================================================
df = load_full_season(2024)

# === Derived Predictors =======================================
df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

def compute_driver_form(sub):
    driver = sub.name[1]
    sub = sub.sort_values("Round").reset_index(drop=True)
    forms = []
    for i in range(len(sub)):
        if i == 0:
            forms.append(prev_driver_form.get(driver, np.nan))
        elif i == 1:
            vals = [sub.loc[i-1, "RacePosition"]]
            if driver in prev_driver_form: vals += [prev_driver_form[driver]] * 2
            forms.append(np.mean(vals))
        elif i == 2:
            vals = list(sub.loc[:i-1, "RacePosition"])
            if driver in prev_driver_form: vals += [prev_driver_form[driver]]
            forms.append(np.mean(vals[-3:]))
        else:
            forms.append(sub.loc[i-3:i-1, "RacePosition"].mean())
    sub["DriverForm"] = forms
    return sub

df = df.groupby(["Year", "Driver"], group_keys=False).apply(compute_driver_form)

def compute_team_strength(sub):
    team = sub.name[1]
    sub = sub.sort_values("Round").reset_index(drop=True)
    prev_seed = prev_team_strength.get(team, np.nan)
    strengths = []
    for i in range(len(sub)):
        if i == 0:
            strengths.append(prev_seed)
        else:
            strengths.append(sub.loc[:i-1, "Points"].mean())
    sub["TeamStrength"] = strengths
    return sub

df = df.groupby(["Year", "Team"], group_keys=False).apply(compute_team_strength)

# === Reliability + Overtake Index =============================
df["DriverDNF_Rate"] = df.groupby("Driver")["DNF"].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df["TeamDNF_Rate"] = df.groupby("Team")["DNF"].transform(lambda x: x.shift().expanding().mean())

overtake_index = {
    "Bahrain Grand Prix":4.5,"Saudi Arabian Grand Prix":4.2,"Australian Grand Prix":3.2,"Japanese Grand Prix":3.0,
    "Chinese Grand Prix":3.3,"Miami Grand Prix":2.8,"Emilia Romagna Grand Prix":2.5,"Monaco Grand Prix":1.0,
    "Canadian Grand Prix":3.8,"Spanish Grand Prix":3.2,"Austrian Grand Prix":4.0,"British Grand Prix":4.2,
    "Hungarian Grand Prix":2.0,"Belgian Grand Prix":4.7,"Dutch Grand Prix":3.5,"Italian Grand Prix":4.6,
    "Azerbaijan Grand Prix":2.2,"Singapore Grand Prix":1.2,"United States Grand Prix":5.0,"Mexican Grand Prix":4.4,
    "Las Vegas Grand Prix":3.8,"Qatar Grand Prix":3.6,"São Paulo Grand Prix":4.3,"Abu Dhabi Grand Prix":3.4
}
df["OvertakeIndex"] = df["Circuit"].map(overtake_index)
df["OvertakeIndex"].fillna(np.mean(list(overtake_index.values())), inplace=True)

# === Final Sort + Export ======================================
df = df.sort_values(["Round", "RacePosition"], na_position="last").reset_index(drop=True)
print(f"\n✅ Final dataset shape: {df.shape}")
print("\n=== Entry counts per circuit ===")
print(df.groupby("Circuit").size())

out_path = r"C:\Users\rohan\Downloads\F1_2024_Full_Dataset_Predictors_DashboardLogic_24Races.xlsx"
df.to_excel(out_path, index=False)
print(f"\n📁 Dataset (24 races, dashboard-compatible) saved to:\n{out_path}")
