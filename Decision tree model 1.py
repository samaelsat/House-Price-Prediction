import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn. tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import tree
import joblib
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

print("="*70)
print("LOADING DATA")
print("="*70)

# Load dataset
df = pd.read_csv('/content/new.csv', encoding='latin-1')  # Added encoding='latin-1'

print(f"\nDataset Shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nBasic Statistics:")
print(df.describe())

# Define target column
target_column = 'totalPrice'  # Adjust if different
print(f"\n✅ Target variable: {target_column}")

print("\n" + "="*70)
print("REMOVING DATA LEAKAGE FEATURES")
print("="*70)

# Features that cause data leakage
leakage_features = ['price', 'id', 'url', 'Cid', 'DOM']

print("\nRemoving these features:")
removed_count = 0
for col in leakage_features: 
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"  ✅ Removed: {col}")
        removed_count += 1

if removed_count == 0:
    print("  ⚠️ No leakage features found in dataset")

print(f"\nDataset shape after removing leakage:  {df.shape}")

print("\n" + "="*70)
print("IDENTIFYING FEATURE TYPES")
print("="*70)

# Separate numerical and categorical
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove target from lists
if target_column in numerical_features: 
    numerical_features.remove(target_column)
if target_column in categorical_features: 
    categorical_features.remove(target_column)

print(f"\n📊 NUMERICAL FEATURES ({len(numerical_features)}):")
for i, col in enumerate(numerical_features, 1):
    print(f"  {i}. {col}")

print(f"\n📝 CATEGORICAL FEATURES ({len(categorical_features)}):")
for i, col in enumerate(categorical_features, 1):
    n_unique = df[col].nunique()
    sample_values = df[col].dropna().unique()[:3]
    print(f"  {i}. {col}: {n_unique} unique values")
    print(f"      Sample:  {sample_values. tolist()}")

print("\n" + "="*70)
print("HANDLING MISSING VALUES")
print("="*70)

# Check missing values
missing_count = df.isnull().sum()
missing_features = missing_count[missing_count > 0]

if len(missing_features) > 0:
    print("\nMissing values found:")
    for col, count in missing_features.items():
        pct = (count / len(df)) * 100
        print(f"  - {col}: {count} ({pct:.2f}%)")

    # Fill numerical with median
    print("\n✅ Filling numerical features with median...")
    for col in numerical_features:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  - {col}:  filled with {median_val:.2f}") # Corrected format specifier

    # Fill categorical with mode or 'Unknown'
    print("\n✅ Filling categorical features with mode...")
    for col in categorical_features:
        if df[col]. isnull().any():
            if len(df[col].mode()) > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  - {col}:  filled with '{mode_val}'")
            else:
                df[col].fillna('Unknown', inplace=True)
                print(f"  - {col}:  filled with 'Unknown'")

    print("\n✅ All missing values handled!")
else:
    print("\n✅ No missing values found!")

print("\n" + "="*70)
print("CATEGORIZING FEATURES FOR ENCODING")
print("="*70)

# Lists for different encoding strategies
one_hot_features = []
label_encoding_features = []
skip_features = []

print("\nAnalyzing each categorical feature:\n")

for col in categorical_features:
    n_unique = df[col].nunique()
    sample_values = df[col].dropna().unique()[:5]
    
    print(f"{col}:")
    print(f"  - Unique values: {n_unique}")
    print(f"  - Sample:  {sample_values.tolist()}")
    
    if n_unique > 50:
        print(f"  → ⚠️ TOO MANY categories - SKIPPING")
        skip_features.append(col)
    elif col in ['renovationCondition']:   # Has natural order
        print(f"  → ✅ LABEL ENCODING (ordinal)")
        label_encoding_features.append(col)
    else:
        print(f"  → ✅ ONE-HOT ENCODING (nominal)")
        one_hot_features.append(col)
    print()

print("="*70)
print("ENCODING STRATEGY SUMMARY:")
print("="*70)
print(f"One-Hot Encoding ({len(one_hot_features)}): {one_hot_features}")
print(f"Label Encoding ({len(label_encoding_features)}): {label_encoding_features}")
print(f"Skipping ({len(skip_features)}): {skip_features}")

print("\n" + "="*70)
print("APPLYING ONE-HOT ENCODING")
print("="*70)

df_encoded = df.copy()

if one_hot_features:
    print("\n✅ Applying One-Hot Encoding:\n")
    
    for col in one_hot_features: 
        if col in df_encoded.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True, dtype=int)
            
            print(f"  {col}:")
            print(f"    - Original categories: {df_encoded[col]. nunique()}")
            print(f"    - Created {len(dummies. columns)} binary features")
            print(f"    - New columns: {dummies.columns.tolist()[:3]}...")
            
            # Add to dataframe
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            # Drop original
            df_encoded = df_encoded.drop(columns=[col])
    
    print(f"\n✅ One-Hot Encoding complete!")
else:
    print("\n⚠️ No features for One-Hot Encoding")

print(f"\nDataset shape after One-Hot Encoding: {df_encoded.shape}")

print("\n" + "="*70)
print("APPLYING LABEL ENCODING")
print("="*70)

if label_encoding_features:
    print("\n✅ Applying Label Encoding:\n")
    
    for col in label_encoding_features:
        if col in df_encoded.columns:
            
            # Special handling for renovationCondition (ordinal)
            if col == 'renovationCondition':
                print(f"  {col} (ordinal encoding):")
                
                # Define order - adjust based on your data
                order_mapping = {
                    'other': 0,
                    'rough': 1,
                    'Simplicity': 2,
                    'hardcover': 3
                }
                
                df_encoded[col + '_encoded'] = df_encoded[col].map(order_mapping)
                df_encoded[col + '_encoded']. fillna(0, inplace=True)
                
                print(f"    - Mapping: {order_mapping}")
            else:
                # Generic label encoding
                print(f"  {col}:")
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col]. astype(str))
                print(f"    - Encoded {df_encoded[col].nunique()} categories")
            
            # Drop original column
            df_encoded = df_encoded.drop(columns=[col])
            print(f"    ✅ Created:  {col}_encoded\n")
    
    print("✅ Label Encoding complete!")
else:
    print("\n⚠️ No features for Label Encoding")

print(f"\nDataset shape after Label Encoding: {df_encoded.shape}")

print("\n" + "="*70)
print("DROPPING HIGH CARDINALITY FEATURES")
print("="*70)

if skip_features:
    print("\n⚠️ Dropping features with too many categories:\n")
    for col in skip_features:
        if col in df_encoded.columns:
            n_unique = df_encoded[col].nunique()
            df_encoded = df_encoded.drop(columns=[col])
            print(f"  ✅ Dropped: {col} ({n_unique} unique values)")
    
    print("\n✅ High cardinality features removed!")
else:
    print("\n✅ No high cardinality features to drop")

print(f"\nDataset shape after dropping: {df_encoded. shape}")

print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Separate features and target
X = df_encoded. drop(columns=[target_column])
y = df_encoded[target_column]

print(f"\nBase features:  {X.shape[1]}")
print(f"Target samples: {len(y)}")

print("\nCreating engineered features:\n")

# 1. Building Age
# Ensure 'constructionTime' is numeric before calculation
if 'constructionTime' in X.columns:
    X['constructionTime'] = pd.to_numeric(X['constructionTime'], errors='coerce')
    current_year = 2017  # Adjust to your data's year range
    X['building_age'] = current_year - X['constructionTime']
    print(f"  ✅ building_age = {current_year} - constructionTime")
    print(f"     Range: {X['building_age'].min():. 0f} to {X['building_age'].max():.0f} years")

# 2. Total Rooms
room_cols = ['livingRoom', 'drawingRoom', 'kitchen', 'bathRoom']
# Ensure room_cols are numeric before sum
for col in room_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0) # Fill NaN after coerce with 0
available_rooms = [col for col in room_cols if col in X.columns]
if available_rooms:
    X['total_rooms'] = X[available_rooms].sum(axis=1)
    print(f"\n  ✅ total_rooms = sum of {len(available_rooms)} room types")
    print(f"     Range: {X['total_rooms'].min():.0f} to {X['total_rooms'].max():.0f} rooms")

# 3. Distance from City Center
if 'Lng' in X.columns and 'Lat' in X.columns:
    center_lng, center_lat = 116.4074, 39.9042  # Beijing center
    X['distance_from_center'] = np.sqrt(
        (X['Lng'] - center_lng)**2 + (X['Lat'] - center_lat)**2
    )
    print(f"\n  ✅ distance_from_center = sqrt((Lng-{center_lng})² + (Lat-{center_lat})²)")
    print(f"     Range: {X['distance_from_center'].min():.4f} to {X['distance_from_center'].max():.4f}")

# 4. Square Meters per Room
if 'square' in X.columns and 'total_rooms' in X.columns:
    X['sqm_per_room'] = X['square'] / (X['total_rooms'] + 1)
    print(f"\n  ✅ sqm_per_room = square / (total_rooms + 1)")
    print(f"     Range: {X['sqm_per_room'].min():.2f} to {X['sqm_per_room'].max():.2f} sqm")

# 5. Location Clusters
if 'Lng' in X.columns and 'Lat' in X.columns:
    coords = X[['Lng', 'Lat']].values
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    X['location_cluster'] = kmeans.fit_predict(coords)
    print(f"\n  ✅ location_cluster = KMeans clustering (15 neighborhoods)")
    print(f"     Clusters: 0 to 14")

# --- FIX: Ensure unique column names to prevent ValueError in later steps ---
# This handles potential duplicate columns created during one-hot encoding or other transformations.
# A more robust approach using a counter.
new_cols = []
cols_count = {}
for col in X.columns:
    original_col = col
    count = 0
    temp_col = original_col
    while temp_col in new_cols:
        count += 1
        temp_col = f"{original_col}_{count}"
    new_cols.append(temp_col)
X.columns = new_cols
# -------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"✅ Feature Engineering Complete!")
print(f"   Total features: {X.shape[1]} (added {X.shape[1] - df_encoded.shape[1] + 1} new features)")
print(f"{'='*70}")

print("\n" + "="*70)
print("CHECKING FOR DATA LEAKAGE (CORRELATION)")
print("="*70)

# Combine for correlation analysis
temp_df = pd.concat([X, y], axis=1)
correlations = temp_df.corr()[target_column]. abs().sort_values(ascending=False)

print("\nTop 15 correlations with target:\n")
for i, (feature, corr) in enumerate(correlations.head(16).items(), 1):
    if feature != target_column:
        if corr > 0.95:
            status = "🚨 DATA LEAKAGE"
        elif corr > 0.80:
            status = "⚠️ High"
        else:
            status = "✅ OK"
        
        bar = '█' * int(corr * 50)
        print(f"  {i: 2d}. {feature:<30} {corr:.4f} {bar}  {status}")

# Remove high correlation features
leakage_cols = correlations[(correlations > 0.95) & (correlations < 1.0)].index.tolist()

if leakage_cols:
    print(f"\n⚠️ WARNING:  Removing {len(leakage_cols)} features with >0.95 correlation:")
    for col in leakage_cols:
        print(f"  - {col}:  {correlations[col]:.4f}")
    X = X.drop(columns=leakage_cols)
    print(f"\n✅ Clean feature matrix:  {X.shape}")
else:
    print("\n✅ No data leakage detected!")

print("\n" + "="*70)
print("REMOVING OUTLIERS")
print("="*70)

# IQR method
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nTarget variable: {target_column}")
print(f"  Q1 (25th percentile): {Q1:.2f}")
print(f"  Q3 (75th percentile): {Q3:.2f}")
print(f"  IQR:  {IQR:.2f}")
print(f"  Lower bound: {lower_bound:.2f}")
print(f"  Upper bound: {upper_bound:.2f}")

# Count outliers
mask = (y >= lower_bound) & (y <= upper_bound)
outliers_count = (~mask).sum()
outliers_pct = (outliers_count / len(y)) * 100

print(f"\nOutliers detected: {outliers_count} ({outliers_pct:.2f}%)")

# Remove outliers
X_clean = X[mask]
y_clean = y[mask]

print(f"\nDataset shape before:  {X. shape}")
print(f"Dataset shape after:   {X_clean.shape}")
print(f"✅ Outliers removed!")

print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean,
    test_size=0.2,
    random_state=42
)

print(f"\nSplit ratio: 80% training, 20% testing")
print(f"\n📊 TRAINING SET:")
print(f"   Samples: {X_train.shape[0]}")
print(f"   Features:  {X_train.shape[1]}")
print(f"   Target range: {y_train.min():.2f} to {y_train. max():.2f}")
print(f"   Target mean: {y_train.mean():.2f}")

print(f"\n📊 TESTING SET:")
print(f"   Samples: {X_test.shape[0]}")
print(f"   Features: {X_test.shape[1]}")
print(f"   Target range: {y_test.min():.2f} to {y_test.max():.2f}")
print(f"   Target mean: {y_test.mean():.2f}")

print(f"\n✅ Data split complete!")

print("\n" + "="*70)
print("BASELINE DECISION TREE (No Constraints)")
print("="*70)

# Create baseline model
dt_baseline = DecisionTreeRegressor(random_state=42)

print("\nTraining baseline model...")
dt_baseline.fit(X_train, y_train)

# Predictions
y_train_pred_base = dt_baseline.predict(X_train)
y_test_pred_base = dt_baseline. predict(X_test)

# Metrics
train_r2_base = r2_score(y_train, y_train_pred_base)
test_r2_base = r2_score(y_test, y_test_pred_base)
test_rmse_base = np. sqrt(mean_squared_error(y_test, y_test_pred_base))
test_mae_base = mean_absolute_error(y_test, y_test_pred_base)
test_mse_base = mean_squared_error(y_test, y_test_pred_base)

print("\n" + "="*70)
print("BASELINE RESULTS")
print("="*70)
print(f"\nTraining R²:       {train_r2_base:.4f}")
print(f"Testing R²:       {test_r2_base:.4f}")
print(f"Overfitting Gap:  {abs(train_r2_base - test_r2_base):.4f}")
print(f"\nTesting RMSE:     {test_rmse_base:.2f}")
print(f"Testing MAE:      {test_mae_base:.2f}")
print(f"Testing MSE:      {test_mse_base:.2f}")

if abs(train_r2_base - test_r2_base) > 0.1:
    print("\n⚠️ WARNING: High overfitting detected!")
else:
    print("\n✅ Overfitting under control")

print("\n" + "="*70)
print("IMPROVED DECISION TREE (With Constraints)")
print("="*70)

# Create improved model
dt_improved = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=45,
    min_samples_leaf=30,
    max_features='sqrt',
    random_state=42
)

print("\nHyperparameters:")
print(f"  max_depth:          {dt_improved.max_depth}")
print(f"  min_samples_split: {dt_improved.min_samples_split}")
print(f"  min_samples_leaf:  {dt_improved.min_samples_leaf}")
print(f"  max_features:      {dt_improved.max_features}")

print("\nTraining improved model...")
dt_improved.fit(X_train, y_train)

# Predictions
y_train_pred_imp = dt_improved.predict(X_train)
y_test_pred_imp = dt_improved.predict(X_test)

# Metrics
train_r2_imp = r2_score(y_train, y_train_pred_imp)
test_r2_imp = r2_score(y_test, y_test_pred_imp)
test_rmse_imp = np.sqrt(mean_squared_error(y_test, y_test_pred_imp))
test_mae_imp = mean_absolute_error(y_test, y_test_pred_imp)
test_mse_imp = mean_squared_error(y_test, y_test_pred_imp)

print("\n" + "="*70)
print("IMPROVED RESULTS")
print("="*70)
print(f"\nTraining R²:      {train_r2_imp:.4f}")
print(f"Testing R²:       {test_r2_imp:.4f}")
print(f"Overfitting Gap:  {abs(train_r2_imp - test_r2_imp):.4f}")
print(f"\nTesting RMSE:     {test_rmse_imp:.2f}")
print(f"Testing MAE:      {test_mae_imp:.2f}")
print(f"Testing MSE:      {test_mse_imp:.2f}")

print("\n" + "-"*70)
print("IMPROVEMENT:")
print(f"  R² change:     {test_r2_imp - test_r2_base:+.4f}")
print(f"  RMSE change:   {test_rmse_imp - test_rmse_base:+.2f}")

print("\n" + "="*70)
print("CROSS-VALIDATION (5-Fold)")
print("="*70)

print("\nPerforming 5-fold cross-validation...")

# R² scores
cv_r2 = cross_val_score(dt_improved, X_train, y_train, cv=5, scoring='r2')

# RMSE scores
cv_mse = -cross_val_score(dt_improved, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(cv_mse)

# MAE scores
cv_mae = -cross_val_score(dt_improved, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)

print(f"\n{'Fold':<10} {'R²':<12} {'RMSE':<12} {'MAE':<12}")
print("-"*70)
for i in range(5):
    print(f"Fold {i+1:<5} {cv_r2[i]: <12.4f} {cv_rmse[i]:<12.2f} {cv_mae[i]:<12.2f}")

print("-"*70)
print(f"{'Mean':<10} {cv_r2.mean():<12.4f} {cv_rmse. mean():<12.2f} {cv_mae.mean():<12.2f}")
print(f"{'Std Dev':<10} {cv_r2.std():<12.4f} {cv_rmse. std():<12.2f} {cv_mae.std():<12.2f}")

print(f"\n✅ Cross-validation complete!")
print(f"   Average R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")

print("\n" + "="*70)
print("HYPERPARAMETER TUNING (GridSearchCV)")
print("="*70)

# Define parameter grid
param_grid = {
    'max_depth': [10, 20, 30, None], # Added max_depth, as it's crucial for Decision Trees
    'min_samples_split': [30, 50, 100],
    'min_samples_leaf': [15, 25, 50],
    'max_features':  ['sqrt', 'log2', None]
}

print("\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}:  {values}")

total_combinations = np.prod([len(v) for v in param_grid. values()])
print(f"\nTotal combinations: {total_combinations}")
print("This will take a few minutes.. .\n")

# Perform grid search
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\n" + "="*70)
print("GRID SEARCH RESULTS")
print("="*70)

print("\nBest parameters found:")
for param, value in grid_search. best_params_.items():
    print(f"  {param}:  {value}")

print(f"\nBest CV R² score: {grid_search.best_score_:.4f}")

# Get best model
dt_best = grid_search.best_estimator_

print("\n✅ Hyperparameter tuning complete!")

print("\n" + "="*70)
print("BEST MODEL EVALUATION")
print("="*70)

# Predictions
y_train_pred_best = dt_best.predict(X_train)
y_test_pred_best = dt_best. predict(X_test)

# Metrics
train_r2_best = r2_score(y_train, y_train_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)
test_rmse_best = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
test_mse_best = mean_squared_error(y_test, y_test_pred_best)

print("\n" + "="*70)
print("BEST MODEL PERFORMANCE")
print("="*70)
print(f"\nTraining R²:      {train_r2_best:.4f}")
print(f"Testing R²:        {test_r2_best:.4f} ({test_r2_best*100:.2f}% variance explained)")
print(f"Overfitting Gap:  {abs(train_r2_best - test_r2_best):.4f}")
print(f"\nTesting RMSE:     {test_rmse_best:.2f}")
print(f"Testing MAE:       {test_mae_best:.2f}")
print(f"Testing MSE:      {test_mse_best:.2f}")

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if test_r2_best >= 0.80:
    print("🎉 EXCELLENT! Model explains >80% of variance")
elif test_r2_best >= 0.70:
    print("✅ VERY GOOD! Model explains >70% of variance")
elif test_r2_best >= 0.60:
    print("✅ GOOD! Model explains >60% of variance")
elif test_r2_best >= 0.50:
    print("⚠️ FAIR. Model explains >50% of variance")
else:
    print("❌ NEEDS IMPROVEMENT. Model explains <50% of variance")

avg_price = y_test.mean()
error_pct = (test_mae_best / avg_price) * 100
print(f"\nAverage prediction error: {error_pct:.2f}% of house price")
print(f"On a {avg_price:.0f} house, expect ±{test_mae_best:.0f} error")

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_best.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print("="*70)
for i, row in feature_importance.head(20).iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"{row['Feature']:<35} {row['Importance']:.4f} {bar}")

# Visualize
plt.figure(figsize=(10, 8))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['Importance'], color='steelblue')
plt.yticks(range(len(top_15)), top_15['Feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved:  'feature_importance.png'")
plt.show()

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)

# Calculate errors
errors = y_test - y_test_pred_best
absolute_errors = np.abs(errors)
percentage_errors = (absolute_errors / y_test) * 100

print("\nError Statistics:")
print("="*70)
print(f"Mean Error:              {errors.mean():.2f}")
print(f"Mean Absolute Error:     {absolute_errors.mean():.2f}")
print(f"Median Absolute Error:   {absolute_errors.median():.2f}")
print(f"Max Absolute Error:      {absolute_errors.max():.2f}")
print(f"Min Absolute Error:      {absolute_errors.min():.2f}")
print(f"Mean Percentage Error:   {percentage_errors. mean():.2f}%")
print(f"Median Percentage Error: {percentage_errors. median():.2f}%")

# Visualize errors
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred_best, alpha=0.5, s=30, color='steelblue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect')
axes[0, 0]. set_xlabel('Actual Price', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Price', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0]. grid(alpha=0.3)

# 2. Residual Plot
axes[0, 1].scatter(y_test_pred_best, errors, alpha=0.5, s=30, color='coral')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1]. set_xlabel('Predicted Price', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3. Error Distribution
axes[1, 0]. hist(errors, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0]. set_xlabel('Error', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 4. Absolute Error
axes[1, 1].hist(absolute_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].axvline(x=absolute_errors.mean(), color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved: 'error_analysis. png'")
plt.show()
