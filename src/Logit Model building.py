# ==============================================================
#  F1 Logistic Regression | Probability of Top 10 Finish
#  (Cross-Season Seeding + Circuit Overtake Difficulty)
# ==============================================================

import fastf1
import pandas as pd
import numpy as np
import os, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# === Cache setup ==============================================
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache(r"C:\\Users\\rohan\\Downloads\\Research STA199\\cache")

# === Load Single Race Data ====================================
def load_race_data(year: int, gp: str):
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load(laps=True, telemetry=False, weather=True)

        results = session.results[["Abbreviation", "TeamName", "GridPosition",
                                   "Position", "Points", "Status"]].copy()
        results.rename(columns={
            "Abbreviation": "Driver",
            "TeamName": "Team",
            "GridPosition": "QualifyingPosition",
            "Position": "RacePosition"
        }, inplace=True)

        results["DNF"] = results["Status"].apply(
            lambda x: 1 if any(term in str(x) for term in ["Ret", "DNF", "DSQ"]) else 0
        )

        # Lap summaries
        laps = session.laps
        if laps.empty:
            print(f"⚠️ No lap data for {gp} {year}, using only results.")
            results["AvgLapTime"] = np.nan
            results["FastestLapTime"] = np.nan
        else:
            laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
            lap_summary = laps.groupby("Driver")["LapTime_s"].agg(
                AvgLapTime="mean", FastestLapTime="min"
            ).reset_index()
            results = results.merge(lap_summary, on="Driver", how="left")

        # Weather
        if not session.weather_data.empty:
            w = session.weather_data
            results["WeatherTemp"] = w["AirTemp"].mean()
            results["WeatherHumidity"] = w["Humidity"].mean()
            results["WeatherTempVar"] = w["AirTemp"].var()
            results["WeatherHumidityVar"] = w["Humidity"].var()
        else:
            results[["WeatherTemp","WeatherHumidity","WeatherTempVar","WeatherHumidityVar"]] = np.nan

        results["Circuit"] = session.event["EventName"]
        results["Year"] = session.event.year
        results["Round"] = session.event["RoundNumber"]
        results["NumLaps"] = getattr(session, "total_laps", np.nan)
        results["SprintWeekend"] = int(session.event["EventFormat"] == "sprint")

        return results

    except Exception as e:
        print(f"❌ Skipping {gp} {year}: {e}")
        return pd.DataFrame()

# === Load full season =========================================
def load_full_season(year: int):
    all_data = []
    schedule = fastf1.get_event_schedule(year)
    for _, ev in schedule.iterrows():
        if ev["EventFormat"] in ["conventional", "sprint"]:
            gp = ev["EventName"]
            df_gp = load_race_data(year, gp)
            if not df_gp.empty:
                all_data.append(df_gp)
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"\n✅ Loaded {df['Circuit'].nunique()} circuits, {len(df)} driver entries total.")
    return df

# === Load current + previous season ===========================
current_year = 2024
prev_year = current_year - 1

df = load_full_season(current_year)

# === Load previous season context for seeding =================
try:
    prev_schedule = fastf1.get_event_schedule(prev_year)
    prev_data = []
    for _, ev in prev_schedule.iterrows():
        if ev["EventFormat"] in ["conventional", "sprint"]:
            sess = fastf1.get_session(prev_year, ev["EventName"], "R")
            sess.load(laps=False, telemetry=False, weather=False)
            res = sess.results[["Abbreviation", "TeamName", "Position", "Points"]].copy()
            res.rename(columns={"Abbreviation": "Driver", "TeamName": "Team"}, inplace=True)
            res["Year"] = prev_year
            res["Round"] = ev["RoundNumber"]
            prev_data.append(res)
    prev_df = pd.concat(prev_data, ignore_index=True)

    prev_driver_form = prev_df.groupby("Driver")["Position"].apply(lambda x: x.tail(3).mean()).to_dict()
    prev_team_strength = prev_df.groupby("Team")["Points"].mean().to_dict()
    print(f"✅ Loaded previous season ({prev_year}) context for seeding.")
except Exception as e:
    print(f"⚠️ Could not load previous season data: {e}")
    prev_driver_form, prev_team_strength = {}, {}

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

# === Reliability metrics ======================================
df = df.sort_values(["Year", "Round"]).reset_index(drop=True)
df["DriverDNF_Rate"] = df.groupby(["Year", "Driver"])["DNF"].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df["TeamDNF_Rate"] = df.groupby(["Year", "Team"])["DNF"].transform(lambda x: x.shift().expanding().mean())
df[["DriverDNF_Rate", "TeamDNF_Rate"]] = df[["DriverDNF_Rate", "TeamDNF_Rate"]].fillna(0)

print("✅ Added reliability features: DriverDNF_Rate, TeamDNF_Rate.")

# === Overtake Index ===========================================
overtake_index = {
    "Bahrain Grand Prix": 4.5, "Saudi Arabian Grand Prix": 4.2, "Australian Grand Prix": 3.2,
    "Japanese Grand Prix": 3.0, "Chinese Grand Prix": 3.3, "Miami Grand Prix": 2.8,
    "Emilia Romagna Grand Prix": 2.5, "Monaco Grand Prix": 1.0, "Canadian Grand Prix": 3.8,
    "Spanish Grand Prix": 3.2, "Austrian Grand Prix": 4.0, "British Grand Prix": 4.2,
    "Hungarian Grand Prix": 2.0, "Belgian Grand Prix": 4.7, "Dutch Grand Prix": 3.5,
    "Italian Grand Prix": 4.6, "Azerbaijan Grand Prix": 2.2, "Singapore Grand Prix": 1.2,
    "United States Grand Prix": 5.0, "Mexican Grand Prix": 4.4, "Las Vegas Grand Prix": 3.8,
    "Qatar Grand Prix": 3.6, "São Paulo Grand Prix": 4.3, "Abu Dhabi Grand Prix": 3.4
}
df["OvertakeIndex"] = df["Circuit"].map(overtake_index)

# === Target ====================================================
df["Top10"] = (df["RacePosition"] <= 10).astype(int)

# === Pre-race features =========================================
pre_race_features = [
    "QualifyingPosition", "TeamStrength", "DriverForm",
    "TeamDNF_Rate", "DriverDNF_Rate",
    "WeatherTemp", "WeatherHumidity", "WeatherTempVar", "WeatherHumidityVar",
    "NumLaps", "SprintWeekend", "OvertakeIndex"
]

# === Clean data ================================================
model_df = df.dropna(subset=["QualifyingPosition"]).copy()

# Context-aware fill
fill_circuit_mean = ["WeatherTemp","WeatherHumidity","WeatherTempVar","WeatherHumidityVar","OvertakeIndex"]
for col in fill_circuit_mean:
    if col in model_df.columns:
        model_df[col] = model_df.groupby("Circuit")[col].transform(lambda x: x.fillna(x.mean()))
        model_df[col] = model_df[col].fillna(model_df[col].mean())

# Fill team/driver averages
for col in ["TeamStrength", "DriverForm"]:
    model_df[col] = model_df.groupby("Driver")[col].transform(lambda x: x.fillna(x.mean()))
    model_df[col] = model_df[col].fillna(model_df[col].mean())

model_df["NumLaps"] = model_df["NumLaps"].fillna(model_df["NumLaps"].median())

print(f"✅ Cleaned dataset shape: {model_df.shape}")

# === Model =====================================================
X = model_df[pre_race_features]
y = model_df["Top10"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, y_train)

y_pred = logit.predict(X_test)
y_prob = logit.predict_proba(X_test)[:, 1]

print("\n=== 🏁 Logistic Model Performance: Top 10 Finish ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0,1], [0,1], "k--")
plt.title("ROC Curve - Probability of Top 10 Finish")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# === Coefficients =============================================
coef_df = pd.DataFrame({
    "Predictor": pre_race_features,
    "Coefficient": logit.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\n=== Logistic Regression Coefficients ===")
print(coef_df.to_string(index=False, float_format="%.4f"))

# === Sample Race Prediction ====================================
first_circuit = model_df["Circuit"].iloc[0]
sample = model_df[model_df["Circuit"] == first_circuit].copy()
sample["Predicted_Prob_Top10"] = logit.predict_proba(scaler.transform(sample[pre_race_features]))[:, 1]

# Sort by finishing position (1–20)
sample = sample.sort_values("RacePosition", ascending=True).reset_index(drop=True)

# Print first 20 entries (or all if fewer)
print(f"\nSample predicted probabilities (Circuit: {first_circuit}) — sorted by finishing position:")
print(
    sample[
        ["RacePosition", "Driver", "Team", "QualifyingPosition", "Predicted_Prob_Top10", "Top10"]
    ].head(20).to_string(index=False, float_format="%.3f")
)