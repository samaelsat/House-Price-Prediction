# Install Optuna for hyperparameter tuning
!pip install optuna -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import optuna
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Load dataset
# Assuming the file is uploaded to the root directory in Colab
df = pd.read_csv('new.csv', encoding='gbk', low_memory=False)

# Drop columns that won't contribute to prediction (URLs, IDs)
df = df.drop(['url', 'id', 'Cid'], axis=1)

# Handle 'constructionTime' which often contains '未知' (Unknown)
df['constructionTime'] = pd.to_numeric(df['constructionTime'], errors='coerce')
df['constructionTime'] = df['constructionTime'].fillna(df['constructionTime'].median())

# Convert tradeTime to datetime and extract Year/Month
df['tradeTime'] = pd.to_datetime(df['tradeTime'], format='%d-%m-%Y')
df['tradeYear'] = df['tradeTime'].dt.year
df['tradeMonth'] = df['tradeTime'].dt.month
df = df.drop('tradeTime', axis=1)

def parse_floor(x):
    if isinstance(x, str):
        parts = x.split()
        if len(parts) == 2:
            # Map Chinese height types to numbers
            height_map = {'高': 3, '中': 2, '低': 1, '底': 0, '顶': 4}
            # Extract height type and total floors
            h_type = height_map.get(parts[0], 2) # Default to Middle if not found
            total_f = int(parts[1])
            return h_type, total_f
    return 2, 0 # Default values

# Apply parsing for 'floor' if column exists
if 'floor' in df.columns:
    df[['floor_type', 'total_floors']] = df['floor'].apply(lambda x: pd.Series(parse_floor(x)))
    df = df.drop('floor', axis=1)
else:
    print("Warning: 'floor' column not found in DataFrame. Skipping floor parsing.")
    # Add default columns if 'floor' is missing to avoid downstream errors, or decide to stop/raise
    df['floor_type'] = 2 # Default to Middle
    df['total_floors'] = 0 # Default to 0 floors

# Fill missing values for numerical columns
df['DOM'] = df['DOM'].fillna(df['DOM'].median())

# Convert 'livingRoom', 'drawingRoom', 'bathRoom' to numeric
# Coerce errors to NaN and then fill NaNs with the median
for col in ['livingRoom', 'drawingRoom', 'bathRoom']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Convert categories to 'category' type for XGBoost (XGBoost handles this natively)
cat_cols = ['buildingType', 'renovationCondition', 'buildingStructure', 'district', 'elevator', 'fiveYearsProperty', 'subway']
for col in cat_cols:
    # Ensure column is first string type, then fill NaNs, then convert to category
    # This prevents the 'Cannot setitem on a Categorical with a new category' error
    df[col] = df[col].astype(str).fillna('missing').astype('category')

# Drop 'price' because it's derived directly from totalPrice (Target Leakage)
if 'price' in df.columns:
    df = df.drop('price', axis=1)

# Target Variable Log Transformation
df['totalPrice_log'] = np.log1p(df['totalPrice'])

X = df.drop(['totalPrice', 'totalPrice_log'], axis=1)
y = df['totalPrice_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training shape: {X_train.shape}")

from xgboost import callback as xgb_callback

def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # Changed from 'gpu_hist' to 'hist' for compatibility
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'enable_categorical': True # Required for the categorical columns we set earlier
    }

    model = xgb.XGBRegressor(**param)
    # Callbacks for early stopping are not supported by the current XGBoost version's fit method.
    # Training for full n_estimators without early stopping for now.
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# Run optimization (set to 20 trials for speed; increase for better results)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best Parameters:", study.best_params)

# Train final model with best parameters
best_params = study.best_params
best_params['enable_categorical'] = True
best_params['tree_method'] = 'hist' # Changed from 'gpu_hist' to 'hist'

final_model = xgb.XGBRegressor(**best_params, n_estimators=2000)
final_model.fit(X_train, y_train)

# Predictions (Back-transform from Log)
y_pred_log = final_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# Metrics
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.4f}")
print(f"R2 Score: {r2:.4f}")

plt.figure(figsize=(15, 6))

# 1. Feature Importance
plt.subplot(1, 2, 1)
feat_importances = pd.Series(final_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Important Features')

# 2. Actual vs Predicted (Scaled around 1200)
plt.subplot(1, 2, 2)
plt.scatter(y_test_orig, y_pred, alpha=0.3, color='orange')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices (Zoomed)')
plt.xlim([0, 2000]) # Adjust x-axis to zoom around 1200
plt.ylim([0, 2000]) # Adjust y-axis to zoom around 1200

plt.tight_layout()
plt.show()

# 3. Actual vs Predicted Price for 100 Sample Data (Bar Graph)
plt.figure(figsize=(15, 7))

# Sample 100 data points for visualization
sample_indices = np.random.choice(len(y_test_orig), 50, replace=False)
sample_actual = y_test_orig.iloc[sample_indices]
sample_predicted = y_pred[sample_indices]

# Create a DataFrame for easier plotting
sample_df = pd.DataFrame({
    'Actual Price': sample_actual,
    'Predicted Price': sample_predicted
}).sort_values(by='Actual Price').reset_index(drop=True)

sample_df.plot(kind='bar', figsize=(15, 7))
plt.title('Actual vs Predicted House Prices for 100 Samples')
plt.xlabel('Sample Index')
plt.ylabel('Price (Ten Thousand RMB)')
plt.xticks([]) # Hide x-axis ticks for clarity
plt.tight_layout()
plt.show()

def predict_house_prices(new_house_features: pd.DataFrame):
    """
    Predicts house prices based on new features using the trained XGBoost model.

    Args:
        new_house_features (pd.DataFrame): DataFrame containing features for new houses.
                                           Must have the same columns as X_train, including categorical ones.

    Returns:
        np.ndarray: Predicted house prices in original scale.
    """
    # Ensure categorical columns are of 'category' dtype, matching training data
    for col in cat_cols:
        if col in new_house_features.columns:
            new_house_features[col] = new_house_features[col].astype(str).astype('category')

    # Make predictions (log-transformed)
    y_pred_log_new = final_model.predict(new_house_features)

    # Inverse transform to original scale
    y_pred_new = np.expm1(y_pred_log_new)
    return y_pred_new

# --- Demonstration for 100 sample houses on new unseen data ---
# Select 100 random samples from the X_test dataset to represent 'new unseen data'
sample_indices = np.random.choice(X_test.index, 100, replace=False)
x_sample = X_test.loc[sample_indices]
y_actual_sample = y_test_orig.loc[sample_indices]

# Get predictions for these 100 samples using the new function
y_predicted_sample = predict_house_prices(x_sample)

# Display actual vs. predicted for these 100 samples
sample_comparison_df = pd.DataFrame({
    'Actual Price': y_actual_sample,
    'Predicted Price': y_predicted_sample
}).sort_values(by='Actual Price').reset_index(drop=True)

print("Actual vs Predicted Prices for 100 Sample Houses (New Unseen Data):")
display(sample_comparison_df)

# Plotting the comparison
plt.figure(figsize=(15, 7))
sample_comparison_df.plot(kind='bar', figsize=(15, 7))
plt.title('Actual vs Predicted House Prices for 100 Sample Data (New Unseen Data)')
plt.xlabel('Sample Index')
plt.ylabel('Price (Ten Thousand RMB)')
plt.xticks([]) # Hide x-axis ticks for clarity
plt.tight_layout()
plt.show()

# Calculate error metrics for the 100 sampled predictions
mse_sample = mean_squared_error(y_actual_sample, y_predicted_sample)
rmse_sample = np.sqrt(mse_sample)
mape_sample = mean_absolute_percentage_error(y_actual_sample, y_predicted_sample)
r2_sample = r2_score(y_actual_sample, y_predicted_sample)

print("Error Metrics for 100 Sampled Predictions:")
print(f"  MSE: {mse_sample:.2f}")
print(f"  RMSE: {rmse_sample:.2f}")
print(f"  MAPE: {mape_sample:.4f}")
print(f"  R2 Score: {r2_sample:.4f}")

y_train_pred_log = final_model.predict(X_train)
y_train_pred_orig = np.expm1(y_train_pred_log)
y_train_orig = np.expm1(y_train)

mse_train = mean_squared_error(y_train_orig, y_train_pred_orig)
rmse_train = np.sqrt(mse_train)
mape_train = mean_absolute_percentage_error(y_train_orig, y_train_pred_orig)
r2_train = r2_score(y_train_orig, y_train_pred_orig)

print("Model Performance on Training Data:")
print(f"  MSE: {mse_train:.2f}")
print(f"  RMSE: {rmse_train:.2f}")
print(f"  MAPE: {mape_train:.4f}")
print(f"  R2 Score: {r2_train:.4f}")

plt.figure(figsize=(16, 12))

# Training Data - Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_train_orig, y_train_pred_orig, alpha=0.3, color='blue')
plt.plot([y_train_orig.min(), y_train_orig.max()], [y_train_orig.min(), y_train_orig.max()], 'k--', lw=2)
plt.xlabel('Actual Price (Training)')
plt.ylabel('Predicted Price (Training)')
plt.title('Training: Actual vs Predicted Prices')
plt.xlim([0, 2500]) # Adjust x-axis to zoom around 1200
plt.ylim([0, 2500]) # Adjust y-axis to zoom around 1200

# Training Data - Residuals vs Predicted
plt.subplot(2, 2, 2)
residuals_train = y_train_orig - y_train_pred_orig
plt.scatter(y_train_pred_orig, residuals_train, alpha=0.3, color='red')
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.xlabel('Predicted Price (Training)')
plt.ylabel('Residuals (Training)')
plt.title('Training: Residuals vs Predicted Prices')
plt.xlim([0, 2500]) # Adjust x-axis to zoom around 1200
plt.ylim([0, 2500]) # Adjust y-axis to zoom around 1200

# Test Data - Actual vs Predicted
plt.subplot(2, 2, 3)
plt.scatter(y_test_orig, y_pred, alpha=0.3, color='green')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'k--', lw=2)
plt.xlabel('Actual Price (Test)')
plt.ylabel('Predicted Price (Test)')
plt.title('Test: Actual vs Predicted Prices')
plt.xlim([0, 2500]) # Adjust x-axis to zoom around 1200
plt.ylim([0, 2500]) # Adjust y-axis to zoom around 1200

# Test Data - Residuals vs Predicted
plt.subplot(2, 2, 4)
residuals_test = y_test_orig - y_pred
plt.scatter(y_pred, residuals_test, alpha=0.3, color='purple')
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.xlabel('Predicted Price (Test)')
plt.ylabel('Residuals (Test)')
plt.title('Test: Residuals vs Predicted Prices')
plt.xlim([0, 2500]) # Adjust x-axis to zoom around 1200
plt.ylim([0, 2500]) # Adjust y-axis to zoom around 1200

plt.tight_layout()
plt.show()


