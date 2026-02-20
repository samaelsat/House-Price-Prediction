import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use("default")

# Load the dataset from CSV (update path/encoding as needed).
df = pd.read_csv("/content/new.csv", encoding='latin1')  # 🔁 update path and encoding

# Define the target variable for prediction.
target_col = "price"   # 🔁 update if needed

print("Initial shape:", df.shape)
df.head()

# Create a working copy for feature engineering.
df_fe = df.copy()

# Derive building age when build year is available.
if "build_year" in df_fe.columns:
    CURRENT_YEAR = 2024
    df_fe["building_age"] = CURRENT_YEAR - df_fe["build_year"]
    df_fe.drop(columns=["build_year"], inplace=True)

# Identify skewed numeric features for log transforms.
skewed_features = [
    col for col in df_fe.columns
    if col not in [target_col]
    and df_fe[col].dtype != "object"
    and df_fe[col].skew() > 1
]

# Reduce skewness with log1p.
for col in skewed_features:
    df_fe[col] = np.log1p(df_fe[col])

# Keep only numeric features for modeling.
df_fe = df_fe.select_dtypes(include=[np.number])
print("After feature engineering:", df_fe.shape)

# Split features and target.
X = df_fe.drop(columns=[target_col])
y = df_fe[target_col]

# Train/test split for evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

from sklearn.impute import SimpleImputer

# Impute missing values and scale features.
imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Baseline linear regression on raw target.
lr_raw = LinearRegression()
lr_raw.fit(X_train_scaled, y_train)

y_train_pred_raw = lr_raw.predict(X_train_scaled)
y_test_pred_raw = lr_raw.predict(X_test_scaled)

# Helper to print evaluation metrics.
def print_metrics(y_true, y_pred, title):
    print(f"\n{title}")
    print("-" * 45)
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("R²  :", r2_score(y_true, y_pred))

# Evaluate baseline model on raw target scale.
print_metrics(y_train, y_train_pred_raw, "Train (Raw Target)")
print_metrics(y_test, y_test_pred_raw, "Test (Raw Target)")

# Diagnostic plots: actual vs predicted and residuals.
def diagnostic_plots(y_true, y_pred, label):
    residuals = y_true - y_pred

    plt.figure(figsize=(14, 5))

    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--" , color="red"
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted ({label})")

    # Residual Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot ({label})")

    plt.tight_layout()
    plt.show()

# Plot diagnostics for baseline model.
diagnostic_plots(y_train, y_train_pred_raw, "Train – Raw Target")
diagnostic_plots(y_test, y_test_pred_raw, "Test – Raw Target")

# Train a model on log-transformed targets.
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

lr_log = LinearRegression()
lr_log.fit(X_train_scaled, y_train_log)

y_train_pred_log = lr_log.predict(X_train_scaled)
y_test_pred_log = lr_log.predict(X_test_scaled)

# Invert log transform for metric reporting.
y_train_pred_log_inv = np.expm1(y_train_pred_log)
y_test_pred_log_inv = np.expm1(y_test_pred_log)

# Evaluate log-target model on original scale.
print_metrics(y_train, y_train_pred_log_inv, "Train (Log Target)")
print_metrics(y_test, y_test_pred_log_inv, "Test (Log Target)")

# Plot diagnostics for log-target model.
diagnostic_plots(y_train, y_train_pred_log_inv, "Train – Log Target")
diagnostic_plots(y_test, y_test_pred_log_inv, "Test – Log Target")

# Visualize a small sample of predictions.
sample_size = 40

comparison_df = pd.DataFrame({
    'Actual': y_test.head(sample_size).values,
    'Predicted': y_test_pred_log_inv[:sample_size]
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
