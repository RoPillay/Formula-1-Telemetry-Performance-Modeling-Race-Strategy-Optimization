import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.stats import spearmanr, kendalltau

# ==============================
# 1. LOAD + PREPROCESS
# ==============================

df = pd.read_excel(
    "C:/Users/Owner/OneDrive/Download1/Research STA199/F1_2024_Full_Dataset_Predictors_DashboardLogic_24Races (1).xlsx"
)

zero_fill = ["TeamStrength","DriverDNF_Rate","TeamDNF_Rate"]
df[zero_fill] = df[zero_fill].fillna(0)

df["DriverForm"] = (
    df.groupby(["Year","Round"])["DriverForm"]
      .transform(lambda x: x.fillna(x.mean()))
)

df["OvertakeIndex"] = (
    df.groupby(["Year","Round"])["OvertakeIndex"]
      .transform(lambda x: x.fillna(x.mean()))
)

df = df.dropna(subset=["QualifyingPosition"])

# ==============================
# 2. QUALIFYING BINS
# ==============================

def bin_quali(pos):
    if pos <= 3:
        return "Front"
    elif pos <= 10:
        return "Mid"
    else:
        return "Back"

df["QualiBin"] = df["QualifyingPosition"].apply(bin_quali)

quali_bin_dummies = pd.get_dummies(
    df["QualiBin"],
    prefix="QualiBin",
    drop_first=True
)

df = pd.concat([df, quali_bin_dummies], axis=1)

continuous_predictors = [
    "DriverForm",
    "TeamStrength",
    "DriverDNF_Rate",
    "TeamDNF_Rate",
    "OvertakeIndex"
]

predictors = list(quali_bin_dummies.columns) + continuous_predictors

model_df = df[["Year","Round","Driver","RacePosition"] + predictors].copy()

# Standardize continuous
scaler = StandardScaler()
model_df[continuous_predictors] = scaler.fit_transform(model_df[continuous_predictors])

# ==============================
# 3. BUILD RACE STRUCTURE
# ==============================

races = []

for (year, round_), g in model_df.groupby(["Year","Round"]):
    g = g.sort_values("RacePosition")
    X = g[predictors].astype(float).to_numpy()
    order = np.arange(len(g))

    races.append({
        "year": year,
        "round": round_,
        "X": X,
        "order": order
    })

print(f"Total races: {len(races)}")

# ==============================
# 4. TRAIN / TEST SPLIT
# ==============================

train_races, test_races = train_test_split(
    races,
    test_size=0.30,
    random_state=42
)

print(f"Train races: {len(train_races)}")
print(f"Test races: {len(test_races)}")

# ==============================
# 5. PL L2 LIKELIHOOD
# ==============================

def pl_neg_log_likelihood_l2(beta, races, lambda_l2=1.0):
    ll = 0.0
    
    for race in races:
        X = race["X"]
        order = race["order"]
        theta = X @ beta
        remaining = list(order)

        for i in order:
            theta_rem = theta[remaining]
            theta_rem -= np.max(theta_rem)
            ll += theta[i] - np.log(np.sum(np.exp(theta_rem)))
            remaining.remove(i)

    penalty = lambda_l2 * np.sum(beta**2)
    return -(ll - penalty)

# ==============================
# 6. TRAIN MODEL
# ==============================

p = len(predictors)
beta0 = np.zeros(p)
lambda_l2 = 1.0

res = minimize(
    pl_neg_log_likelihood_l2,
    beta0,
    args=(train_races, lambda_l2),
    method="BFGS"
)

beta_hat = res.x

print("Model trained on training races.")

# ==============================
# 7. TEST EVALUATION
# ==============================

# Test log-likelihood (no penalty!)
def pl_loglik(beta, races):
    ll = 0.0
    for race in races:
        X = race["X"]
        order = race["order"]
        theta = X @ beta
        remaining = list(order)

        for i in order:
            theta_rem = theta[remaining]
            theta_rem -= np.max(theta_rem)
            ll += theta[i] - np.log(np.sum(np.exp(theta_rem)))
            remaining.remove(i)

    return ll

test_loglik = pl_loglik(beta_hat, test_races)
print("Test Log-Likelihood:", test_loglik)

# Rank metrics
spearman_scores = []
kendall_scores = []
top3_scores = []

for race in test_races:
    X = race["X"]
    true_order = race["order"]
    pred_order = np.argsort(-(X @ beta_hat))

    spearman_scores.append(
        spearmanr(true_order, pred_order).correlation
    )

    kendall_scores.append(
        kendalltau(true_order, pred_order).correlation
    )

    pred_top3 = set(pred_order[:3])
    true_top3 = set(true_order[:3])
    top3_scores.append(len(pred_top3 & true_top3)/3)

print("Test Mean Spearman:", np.nanmean(spearman_scores))
print("Test Mean Kendall:", np.nanmean(kendall_scores))
print("Test Top-3 Accuracy:", np.mean(top3_scores))