import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, linear_reset
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------
FILE_PATH = "weather.csv"        # <-- change this
TIME_COLUMN = "date"                  # <-- change if needed
LJUNG_BOX_LAG = 10

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(FILE_PATH)

# Parse time column
df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
df = df.dropna(subset=[TIME_COLUMN])
df = df.set_index(TIME_COLUMN).sort_index()

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

if df.shape[1] == 0:
    raise ValueError("No numeric variables found for analysis.")

print(f"Analyzing {df.shape[1]} variables")
print("-" * 60)

# --------------------------------------------------
# TEST FUNCTIONS
# --------------------------------------------------


def check_normality(series):
    """Shapiro-Wilk + Jarque-Bera"""
    series = series.dropna()
    if len(series) < 20:
        return {
            "Shapiro_p": np.nan,
            "Jarque_Bera_p": np.nan,
            "Normal": False
        }

    shapiro_stat, shapiro_p = stats.shapiro(series)
    jb_stat, jb_p = stats.jarque_bera(series)

    return {
        "Shapiro_p": shapiro_p,
        "Jarque_Bera_p": jb_p,
        "Normal": shapiro_p > 0.05 and jb_p > 0.05
    }

def check_stationarity(series):
    """Augmented Dickey-Fuller test"""
    series = series.dropna()
    if len(series) < 20:
        return {"ADF_p": np.nan, "Stationary": False}

    adf = adfuller(series, autolag="AIC")
    return {
        "ADF_p": adf[1],
        "Stationary": adf[1] < 0.05
    }

def check_linearity(series):
    """Linearity vs time (Ramsey RESET)"""
    series = series.dropna()
    if len(series) < 20:
        return {"RESET_p": np.nan, "Linear": False}

    y = series.values
    X = np.arange(len(series))
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    reset = linear_reset(model, power=2, use_f=True)

    return {
        "RESET_p": reset.pvalue,
        "Linear": reset.pvalue > 0.05
    }

def check_autocorrelation(series):
    """Durbin-Watson + Ljung-Box"""
    series = series.dropna()
    if len(series) < 20:
        return {
            "Durbin_Watson": np.nan,
            "LjungBox_p": np.nan,
            "No_Autocorrelation": False
        }

    dw = durbin_watson(series)
    lb = acorr_ljungbox(series, lags=[LJUNG_BOX_LAG], return_df=True)

    return {
        "Durbin_Watson": dw,
        "LjungBox_p": lb["lb_pvalue"].iloc[0],
        "No_Autocorrelation": lb["lb_pvalue"].iloc[0] > 0.05
    }

# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------

results = {}

for col in df.columns:
    series = df[col]

    results[col] = {}

    results[col].update(check_stationarity(series))
    results[col].update(check_normality(series))
    results[col].update(check_linearity(series))
    results[col].update(check_autocorrelation(series))

results_df = pd.DataFrame(results).T
print(results_df)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
results_df.to_csv("time_series_diagnostics_results.csv")
