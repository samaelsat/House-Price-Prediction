import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)

BASE_DIR = os.getcwd()
DATA_PATHS = [
    os.path.join(BASE_DIR, "archive", "new.csv"),
    os.path.join(BASE_DIR, "new.csv")
]

csv_path = next((p for p in DATA_PATHS if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("❌ new.csv not found")

df = pd.read_csv(csv_path, encoding="gbk")
print("Dataset shape:", df.shape)

df.head()

def process_floor(value):
    if pd.isna(value):
        return np.nan, np.nan

    value = str(value)

    if "高" in value:
        level = 3
    elif "中" in value:
        level = 2
    elif "低" in value or "底" in value:
        level = 1
    else:
        level = 0

    nums = re.findall(r"\d+", value)
    floor_num = int(nums[0]) if nums else np.nan

    return level, floor_num

df[["floor_level", "floor_number"]] = (
    df["floor"].apply(process_floor).apply(pd.Series)
)

df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")

df["trade_year"] = df["tradeTime"].dt.year
df["trade_month"] = df["tradeTime"].dt.month
df["trade_quarter"] = df["tradeTime"].dt.quarter
df["trade_weekday"] = df["tradeTime"].dt.dayofweek

df["constructionTime"] = pd.to_numeric(df["constructionTime"], errors="coerce")
df["building_age"] = df["trade_year"] - df["constructionTime"]
df.loc[df["building_age"] < 0, "building_age"] = np.nan

for col in ["livingRoom", "drawingRoom", "bathRoom", "kitchen"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["total_rooms"] = (
    df["livingRoom"] +
    df["drawingRoom"] +
    df["bathRoom"] +
    df["kitchen"]
)

df["living_ratio"] = df["livingRoom"] / df["total_rooms"]
df["bathroom_ratio"] = df["bathRoom"] / df["total_rooms"]
df["square_per_room"] = df["square"] / df["total_rooms"]

# ⚠️ Potential leakage feature — keep for experiment, remove for final model
df["price_per_room"] = df["totalPrice"] / df["total_rooms"]

CITY_LNG, CITY_LAT = 116.4074, 39.9042

df["distance_from_center"] = np.sqrt(
    (df["Lng"] - CITY_LNG) ** 2 +
    (df["Lat"] - CITY_LAT) ** 2
)

df["DOM_category"] = pd.cut(
    df["DOM"],
    bins=[0, 30, 60, 90, np.inf],
    labels=["Quick", "Normal", "Slow", "Very_Slow"]
)

df["is_luxury"] = (
    (df["price"] > df["price"].quantile(0.75)) &
    (df["renovationCondition"] == 4)
).astype(int)

df["DOM_category"] = pd.cut(
    df["DOM"],
    bins=[0, 30, 60, 90, np.inf],
    labels=["Quick", "Normal", "Slow", "Very_Slow"]
)

df["is_luxury"] = (
    (df["price"] > df["price"].quantile(0.75)) &
    (df["renovationCondition"] == 4)
).astype(int)

LEAKAGE_COLS = [
    "price_per_room"  # direct target leakage
]

df = df.drop(columns=LEAKAGE_COLS, errors="ignore")

print("Leakage features removed ✅")

TARGET = "totalPrice"

X = df.drop(columns=[TARGET, "price"], errors="ignore")
y = df[TARGET]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

Q1, Q3 = y.quantile([0.25, 0.75])
IQR = Q3 - Q1

mask = (y >= Q1 - 3 * IQR) & (y <= Q3 + 3 * IQR)

X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_clean = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_clean.fit(X_train, y_train)

train_pred_clean = rf_clean.predict(X_train)
test_pred_clean = rf_clean.predict(X_test)

def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{label} RMSE: {rmse:.2f}")
    print(f"{label} MAE: {mae:.2f}")
    print(f"{label} R2 Score: {r2:.2f}")

evaluate(y_train, train_pred_clean, "Training (No Leakage)")
evaluate(y_test, test_pred_clean, "Test (No Leakage)")

plt.figure(figsize=(7, 7))
plt.scatter(y_test, test_pred_clean, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (No Leakage)")
plt.tight_layout()
plt.show()

residuals_clean = y_test - test_pred_clean

plt.figure(figsize=(7, 6))
sns.histplot(residuals_clean, bins=50, kde=True)
plt.title("Residual Distribution (No Leakage)")
plt.xlabel("Residual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(test_pred_clean, residuals_clean, alpha=0.4)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted (No Leakage)")
plt.tight_layout()
plt.show()

feat_imp_clean = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_clean.feature_importances_
}).sort_values("importance", ascending=False)

feat_imp_clean.head(15)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_clean["feature"][:15], feat_imp_clean["importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Features (No Leakage)")
plt.tight_layout()
plt.show()

cv_rmse_clean = np.sqrt(
    -cross_val_score(
        rf_clean,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
)

print("CV RMSE:", cv_rmse_clean)
print("Mean CV RMSE:", cv_rmse_clean.mean())

sample_size = 40

comparison_df = pd.DataFrame({
    'Actual': y_test.head(sample_size).values,
    'Predicted': test_pred_clean[:sample_size]
})

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))

x_pos = np.arange(sample_size)
width = 0.35

# Bar chart
axes[0].bar(x_pos - width/2, comparison_df['Actual'].values, width,
           label='Actual Price', alpha=0.8, color='steelblue', edgecolor='black')
axes[0].bar(x_pos + width/2, comparison_df['Predicted'].values, width,
           label='Predicted Price', alpha=0.8, color='coral', edgecolor='black')
axes[0].set_xlabel('Sample Index', fontsize=12)
axes[0].set_ylabel('Price', fontsize=12)
axes[0].set_title(f'Actual vs Predicted Prices (Sample of {sample_size})',
                 fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(range(1, sample_size + 1), fontsize=8)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Line plot
axes[1].plot(range(1, sample_size + 1), comparison_df['Actual'].values,
            'o-', label='Actual Price', linewidth=2, markersize=8, color='steelblue')
axes[1].plot(range(1, sample_size + 1), comparison_df['Predicted'].values,
            's-', label='Predicted Price', linewidth=2, markersize=8, color='coral')
axes[1].set_xlabel('Sample Index', fontsize=12)
axes[1].set_ylabel('Price', fontsize=12)
axes[1].set_title(f'Actual vs Predicted Prices - Line Plot (Sample of {sample_size})',
                 fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('50_sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Sample predictions visualization complete!")
print("="*80)

