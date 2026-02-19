import pandas as pd

# Load raw data (DO NOT manually edit the XLSX)
df = pd.read_excel(
    "C:/Users/Owner/OneDrive/Download1/Research STA199/F1_2024_Full_Dataset_Predictors_DashboardLogic_24Races (1).xlsx"
)

# ------------------Part 1---------------------------

# Structural zeros: team strength & DNF rates
# These variables are defined such that 0 is a valid, meaningful value
# (e.g., no prior DNFs, no points accumulated yet).
zero_fill = [
    "TeamStrength",
    "DriverDNF_Rate",
    "TeamDNF_Rate"
]

df[zero_fill] = df[zero_fill].fillna(0)

# Driver form: neutral baseline for replacements
# Missing DriverForm indicates no prior race history (e.g., replacements).
# We impute using the race-level mean to avoid imposing artificial
# driver ability while allowing team strength and qualifying to dominate.
df["DriverForm"] = (
    df.groupby(["Year", "Round"])["DriverForm"]
      .transform(lambda x: x.fillna(x.mean()))
)

# Overtake difficulty index (track-level variable)
# Filled at the race level to ensure consistency across drivers.
df["OvertakeIndex"] = (
    df.groupby(["Year", "Round"])["OvertakeIndex"]
      .transform(lambda x: x.fillna(x.mean()))
)


# Drop rows missing essential predictors
# Qualifying position must exist for all modeled drivers.
df = df.dropna(subset=["QualifyingPosition"])

# Qualifying position (dummy variable)
quali_dummies = pd.get_dummies(
    df["QualifyingPosition"],
    prefix="Quali",
    drop_first=True  
)

df = pd.concat([df, quali_dummies], axis=1)

quali_cols = list(quali_dummies.columns)

# Final predictor list for Plackett–Luce
continuous_predictors = [
    "DriverForm",
    "TeamStrength",
    "DriverDNF_Rate",
    "TeamDNF_Rate",
    "OvertakeIndex"
]

predictors = quali_cols + continuous_predictors

# Final modeling dataframe

model_df = df[
    ["Year", "Round", "Driver", "RacePosition"] + predictors
].copy()

print("Preprocessing complete.")
print("Missing values by predictor:")
print(model_df[predictors].isna().sum())
#----------------------------------------------------


# ------------------Part 2---------------------------

# Standardizing so we can directly compare coefficients
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

model_df[continuous_predictors] = scaler.fit_transform(model_df[continuous_predictors])
#----------------------------------------------------


# ------------------Part 3---------------------------

# Building race-level rankings
import numpy as np

races = []

for (year, round_), g in model_df.groupby(["Year", "Round"]):
    # Sort drivers by actual race result
    g = g.sort_values("RacePosition")
    
    # Design matrix for this race
    X = g[predictors].astype(float).to_numpy()
    
    # Order is implicit since sorted
    order = np.arange(len(g))
    
    races.append({
        "year": year,
        "round": round_,
        "X": X,
        "order": order,
        "drivers": g["Driver"].values
    })

print(f"Constructed {len(races)} race rankings.")
#----------------------------------------------------


# ------------------Part 4---------------------------

# Defining Plackett-Luce log-likelihood
def pl_neg_log_likelihood(beta, races):
    ll = 0.0
    
    for race in races:
        X = race["X"]
        order = race["order"]
        
        theta = X @ beta
        remaining = list(order)
        
        for i in order:
            theta_rem = theta[remaining]
            theta_rem = theta_rem - np.max(theta_rem)  # stability
            ll += theta[i] - np.log(np.sum(np.exp(theta_rem)))
            remaining.remove(i)
    
    return -ll
#----------------------------------------------------


# ------------------Part 5---------------------------

# Model Fitting
from scipy.optimize import minimize

p = len(predictors)
beta0 = np.zeros(p)

res = minimize(
    pl_neg_log_likelihood,
    beta0,
    args=(races,),
    method="BFGS"
)

beta_hat = res.x
#----------------------------------------------------


# ------------------Part 6---------------------------

# Coefficient interpretation
import pandas as pd
import numpy as np
from scipy.stats import norm

cov = res.hess_inv
se = np.sqrt(np.diag(cov))
z = beta_hat / se
p = 2 * (1 - norm.cdf(np.abs(z)))

results = pd.DataFrame({
    "Predictor": predictors,
    "Coefficient": beta_hat,
    "StdError": se,
    "z_value": z,
    "p_value": p
})

print(results)
#----------------------------------------------------


# ------------------Part 7---------------------------

# Model Evaluation

# a) Log-likelihood
loglik = -pl_neg_log_likelihood(beta_hat, races)
print("Log-likelihood:", loglik)

# b) Rank Correlation
from scipy.stats import spearmanr, kendalltau

spearman_scores = []
kendall_scores = []

for race in races:
    X = race["X"]
    true_order = race["order"]
    pred_order = np.argsort(-(X @ beta_hat))
    
    spearman_scores.append(spearmanr(true_order, pred_order).correlation)
    kendall_scores.append(kendalltau(true_order, pred_order).correlation)

print("Mean Spearman:", np.nanmean(spearman_scores))
print("Mean Kendall:", np.nanmean(kendall_scores))

# c) Top-k accuracy
top3_acc = []

for race in races:
    X = race["X"]
    pred_top3 = set(np.argsort(-(X @ beta_hat))[:3])
    true_top3 = set(race["order"][:3])
    
    top3_acc.append(len(pred_top3 & true_top3) / 3)

print("Top-3 accuracy:", np.mean(top3_acc))
#----------------------------------------------------

# ------------------Part 8---------------------------

race_driver_rows = []

for race in races:
    X = race["X"]
    drivers = race["drivers"]
    order = race["order"]
    
    theta = X @ beta_hat
    exp_theta = np.exp(theta)
    win_probs = exp_theta / np.sum(exp_theta)
    
    # Predicted ranking
    pred_rank = np.argsort(-theta)
    
    # Race-level log-likelihood
    remaining = list(order)
    ll_race = 0.0
    for i in order:
        ll_race += theta[i] - np.log(np.sum(np.exp(theta[remaining])))
        remaining.remove(i)
    
    for idx, driver in enumerate(drivers):
        race_driver_rows.append({
            "Year": race["year"],
            "Round": race["round"],
            "Driver": driver,
            "ActualRacePosition": idx + 1,  # valid due to prior sorting
            "PredictedRank": np.where(pred_rank == idx)[0][0] + 1,
            "PL_Win_Probability": win_probs[idx],
            "Race_LogLikelihood": ll_race
        })

race_driver_df = pd.DataFrame(race_driver_rows)

# Approximate Top-3 probability (clearly labeled)
race_driver_df["Approx_PL_Top3_Probability"] = (
    race_driver_df["PL_Win_Probability"] * 3
).clip(upper=1)

with pd.ExcelWriter("PL_Race_Level_Probabilities_2024(3).xlsx", engine="openpyxl") as writer:
    race_driver_df.to_excel(
        writer,
        sheet_name="Race_Driver_Probabilities",
        index=False
    )

print("Race-level PL probabilities exported.")
#----------------------------------------------------


# ------------------Part 9---------------------------

# Predicted Rank vs Actual Rank
import plotly.express as px
import matplotlib.pyplot as plt

fig = px.scatter(
    race_driver_df,
    x="ActualRacePosition",
    y="PredictedRank",
    color="Round",
    hover_data=["Driver"],
    title="Predicted vs Actual Finishing Position (All Races)"
)

fig.add_shape(
    type="line",
    x0=1, y0=1, x1=20, y1=20,
    line=dict(dash="dash")
)

fig.write_html("Predicted_vs_Actual_All_Races.html")

# Top-3 prediction accuracy by race
race_top3 = []

for race in races:
    X = race["X"]
    true_top3 = set(race["order"][:3])
    pred_top3 = set(np.argsort(-(X @ beta_hat))[:3])
    race_top3.append(len(true_top3 & pred_top3) / 3)

plt.figure(figsize=(10, 4))
plt.bar(range(len(race_top3)), race_top3)
plt.axhline(np.mean(race_top3), linestyle="--")
plt.xlabel("Race")
plt.ylabel("Top-3 Accuracy")
plt.title("Top-3 Prediction Accuracy by Race")
plt.tight_layout()
plt.show()
#----------------------------------------------------
